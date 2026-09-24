#!/usr/bin/env python3
"""Measure forget-document gradients with respect to effective dense LoRA updates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import tempfile
from pathlib import Path

STEPS = (5, 10, 15, 20, 25, 30, 32)
METHODS = ("cb", "npo")
DATASET = "cais/wmdp-bio-forget-corpus"
BASE_MODEL = "EleutherAI/deep-ignorance-unfiltered"
SAMPLE_COUNT = 16
SEED = 42
MAX_TOKENS = 512


def checkpoint_jobs(artifacts: Path):
    """Yield the four trajectories in plot order, rejecting incomplete runs."""
    jobs = []
    for method in METHODS:
        unlearn = artifacts / "models" / f"di-6.9b-wmdp-bio-unlearn-{method}"
        for stage in ("unlearn", "relearn"):
            adapter = unlearn if stage == "unlearn" else Path(f"{unlearn}-relearn")
            base = BASE_MODEL if stage == "unlearn" else str(unlearn / "merged")
            if stage == "relearn" and not Path(base).is_dir():
                raise FileNotFoundError(f"missing merged base model: {base}")
            for step in STEPS:
                checkpoint = adapter / f"checkpoint-{step}"
                if not (checkpoint / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"missing LoRA checkpoint: {checkpoint}")
                jobs.append((method, stage, step, base, checkpoint))
    return jobs


def select_documents(tokenizer):
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train[:1024]")
    if len(dataset) < SAMPLE_COUNT:
        raise ValueError(f"{DATASET} has fewer than {SAMPLE_COUNT} training documents")
    indices = random.Random(SEED).sample(range(len(dataset)), SAMPLE_COUNT)
    documents = []
    for index in indices:
        row = dataset[index]
        fields = ("title", "abstract", "text")
        if any(not isinstance(row.get(field), str) for field in fields):
            raise ValueError(f"document {index} lacks title, abstract, or text")
        text = "\n\n".join(row[field] for field in fields)
        ids = tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
        if len(ids) < 2:
            raise ValueError(f"document {index} has fewer than two tokens")
        documents.append(ids)
    return indices, documents


def ghost_inner_product(left, right):
    """Frobenius inner product of (G.T @ X) without forming a dense update."""
    left_x, left_g = left
    right_x, right_g = right
    return ((left_x @ right_x.T) * (left_g @ right_g.T)).sum()


def mean_cosine_from_factors(directory: Path, layer_count: int, sample_count: int):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    products = torch.zeros(
        (sample_count, sample_count), dtype=torch.float64, device=device
    )
    for layer in range(layer_count):
        factors = [
            tuple(
                factor.to(device)
                for factor in torch.load(
                    directory / f"{layer}-{sample}.pt", weights_only=True
                )
            )
            for sample in range(sample_count)
        ]
        for i in range(sample_count):
            for j in range(i, sample_count):
                value = ghost_inner_product(factors[i], factors[j])
                products[i, j] += value
                if i != j:
                    products[j, i] += value
    norms = products.diag()
    if (
        not torch.isfinite(products).all()
        or not torch.isfinite(norms).all()
        or (norms <= 0).any()
    ):
        raise ValueError(
            "dense-update gradients have zero or non-finite norms/products"
        )
    cosines = products / torch.sqrt(norms[:, None] * norms[None, :])
    pairs = sample_count * (sample_count - 1) // 2
    mean = cosines.triu(diagonal=1).sum().item() / pairs
    if not math.isfinite(mean):
        raise ValueError("mean dense-update gradient cosine is non-finite")
    return mean


def measure_checkpoint(
    base: str, checkpoint: Path, documents: list[list[int]]
) -> float:
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    ]
    if not layers:
        raise ValueError(f"{checkpoint}: no adapted linear layers found")
    for name, module in layers:
        if not hasattr(module, "lora_A") or not module.lora_A:
            raise ValueError(f"{checkpoint}: unsupported adapted layer {name}")

    captured = {}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor) or not output.requires_grad:
                raise ValueError(f"{name}: adapted output has no gradient")
            captured[name] = [inputs[0].detach().squeeze(0).float().cpu(), None]

            def save_gradient(gradient):
                captured[name][1] = gradient.detach().squeeze(0).float().cpu()

            output.register_hook(save_gradient)

        return hook

    for name, module in layers:
        handles.append(module.register_forward_hook(capture(name)))

    try:
        with tempfile.TemporaryDirectory(
            prefix="dense-grad-cosim-", dir=checkpoint.parent
        ) as temporary:
            directory = Path(temporary)
            device = model.get_input_embeddings().weight.device
            for sample, ids in enumerate(documents):
                captured.clear()
                model.zero_grad(set_to_none=True)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(input_ids=input_ids, use_cache=False).logits
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    input_ids[:, 1:].reshape(-1),
                    reduction="mean",
                )
                loss.backward()
                if len(captured) != len(layers):
                    raise ValueError(f"{checkpoint}: missing adapted layer activations")
                for layer, (name, _) in enumerate(layers):
                    x, g = captured[name]
                    if g is None:
                        raise ValueError(
                            f"{checkpoint}: missing output gradient for {name}"
                        )
                    torch.save((x, g), directory / f"{layer}-{sample}.pt")
                del logits, loss, input_ids
            return mean_cosine_from_factors(directory, len(layers), len(documents))
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect_rows(jobs, documents, measure=measure_checkpoint):
    rows = []
    for method, stage, step, base, checkpoint in jobs:
        print(f"measuring {method} {stage} checkpoint-{step}", flush=True)
        rows.append(
            {
                "method": method,
                "stage": stage,
                "checkpoint_step": step,
                "cumulative_step": step + (32 if stage == "relearn" else 0),
                "mean_cosine": measure(base, checkpoint, documents),
                "sample_count": len(documents),
                "pair_count": len(documents) * (len(documents) - 1) // 2,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    jobs = checkpoint_jobs(args.artifacts)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    indices, documents = select_documents(tokenizer)
    rows = collect_rows(jobs, documents)
    output = args.artifacts / "analysis" / "per_sample_param_grad_cosim.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "probe": {
                    "dataset": DATASET,
                    "split": "train[:1024]",
                    "sample_indices": indices,
                    "seed": SEED,
                    "max_tokens": MAX_TOKENS,
                    "loss": "token-mean causal LM cross-entropy",
                    "gradient": "effective dense LoRA update per adapted linear layer",
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
