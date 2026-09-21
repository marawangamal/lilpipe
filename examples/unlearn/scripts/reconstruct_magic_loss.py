#!/usr/bin/env python3
"""Reconstruct training loss curves from MAGIC FSDP checkpoints.

For each checkpoint directory, loads model parameters by merging FSDP
shards, computes cross-entropy loss on a fixed subset of the training
dataset, and saves step -> loss to a CSV.

Requires the bergson3 venv (torch with DTensor support, torchopt).

Usage:
    python -m unlearn.scripts.reconstruct_magic_loss \
        --ckpt_dir /projects/a6a/public/lucia/magic_ultrachat_msl1024_ckpts \
        --dataset ultrachat

    # Run all four checkpoint dirs:
    python -m unlearn.scripts.reconstruct_magic_loss --all
"""

import csv
import os
import re
import time
from dataclasses import dataclass

import torch
from datasets import load_dataset
from simple_parsing import ArgumentParser
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
CKPT_BASE = "/projects/a6a/public/lucia"
MAX_SEQ_LEN = 1024

DATASET_CONFIGS = {
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "text_key": "text",
        "preprocess": lambda ds: ds.map(
            lambda x: {"text": "\n".join(m["content"] for m in x["messages"])}
        ),
    },
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-103-v1",
        "split": "train",
        "text_key": "text",
    },
    "wmdp_retain": {
        "path": "cais/wmdp-corpora",
        "name": "bio-retain-corpus",
        "split": "train",
        "text_key": "text",
    },
    "wmdp_lie_o": {
        "path": "Unlearning/wmdp-lie-o-deep-fried",
        "split": "train",
        "text_key": "text",
    },
}


@dataclass
class ReconstructConfig:
    ckpt_dir: str = ""
    """Checkpoint directory
    (e.g. /projects/a6a/public/lucia/magic_ultrachat_msl1024_ckpts)."""

    dataset: str = ""
    """Dataset name: ultrachat, wikitext, wmdp_retain, wmdp_lie_o."""

    output_dir: str = "runs/magic_loss_curves"
    """Output directory for CSV files."""

    step_interval: int = 25
    """Evaluate every N steps."""

    n_eval_examples: int = 64
    """Number of examples in the fixed eval set."""

    eval_batch_size: int = 8
    """Batch size for forward passes."""

    all: bool = False
    """Process all four magic checkpoint directories."""


def sorted_checkpoints(folder: str) -> list[tuple[int, str]]:
    pattern = re.compile(r"step_(\d+)\.ckpt$")
    checkpoints = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            match = pattern.match(name)
            if match:
                checkpoints.append((int(match.group(1)), path))
    return sorted(checkpoints, key=lambda x: x[0])


def detect_n_shards(ckpt_path: str) -> int:
    return len(
        [
            f
            for f in os.listdir(ckpt_path)
            if f.startswith("rank_") and f.endswith(".shard")
        ]
    )


def load_merged_params(ckpt_path: str, n_shards: int) -> dict[str, torch.Tensor]:
    """Load all rank shards and merge params by concatenating along dim 0."""
    all_params = []
    for rank in range(n_shards):
        shard_path = os.path.join(ckpt_path, f"rank_{rank}.shard")
        shard = torch.load(shard_path, weights_only=False, map_location="cpu")
        params = {}
        for k, v in shard["params"].items():
            if isinstance(v, DTensor):
                v = v.to_local()
            params[k] = v
        all_params.append(params)
        del shard

    merged = {}
    for k in all_params[0]:
        tensors = [s[k] for s in all_params]
        if tensors[0].is_floating_point() and tensors[0].ndim >= 1:
            merged[k] = torch.cat(tensors, dim=0)
        else:
            merged[k] = tensors[0]
    del all_params
    return merged


def ckpt_key_to_hf_key(ckpt_key: str) -> str:
    """Map bergson FSDP param name to HF model param name.

    'gpt_neox.layers.0.attention.query_key_value.parametrizations.weight.original'
    -> 'gpt_neox.layers.0.attention.query_key_value.weight'
    """
    return re.sub(r"\.parametrizations\.(\w+)\.original", r".\1", ckpt_key)


def load_dataset_for_eval(dataset_name: str, n_examples: int) -> tuple[list[str], str]:
    """Load and prepare a fixed eval subset from the training dataset."""
    cfg = DATASET_CONFIGS[dataset_name]

    load_kwargs = {"path": cfg["path"], "split": cfg["split"]}
    if "name" in cfg:
        load_kwargs["name"] = cfg["name"]

    ds = load_dataset(**load_kwargs)

    if "preprocess" in cfg:
        ds = cfg["preprocess"](ds)

    text_key = cfg["text_key"]
    ds = ds.filter(lambda x: len(x[text_key].strip()) > 100)
    ds = ds.select(range(min(n_examples, len(ds))))
    texts = [ds[i][text_key] for i in range(len(ds))]
    return texts, text_key


@torch.no_grad()
def compute_loss(
    model: torch.nn.Module,
    hf_state_dict: dict[str, torch.Tensor],
    tokenizer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute average cross-entropy loss on texts using the given params."""
    model.load_state_dict(hf_state_dict, strict=True)

    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        total_loss += outputs.loss.item()
        n_batches += 1

    return total_loss / n_batches


def detect_dataset_from_ckpt_dir(ckpt_dir: str) -> str:
    dirname = os.path.basename(ckpt_dir)
    for ds_name in ["ultrachat", "wikitext", "wmdp_retain", "wmdp_lie_o"]:
        if ds_name in dirname:
            return ds_name
    raise ValueError(f"Cannot detect dataset from checkpoint dir name: {dirname}")


def process_ckpt_dir(
    ckpt_dir: str,
    dataset_name: str,
    run_cfg: ReconstructConfig,
):
    print(f"\nProcessing: {ckpt_dir}")
    print(f"Dataset: {dataset_name}")

    os.makedirs(run_cfg.output_dir, exist_ok=True)
    csv_path = os.path.join(run_cfg.output_dir, f"loss_curve_{dataset_name}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load eval dataset
    print(f"Loading eval dataset ({run_cfg.n_eval_examples} examples)...")
    texts, _ = load_dataset_for_eval(dataset_name, run_cfg.n_eval_examples)
    print(f"Loaded {len(texts)} examples")

    # Get checkpoint list
    all_ckpts = sorted_checkpoints(ckpt_dir)
    n_shards = detect_n_shards(all_ckpts[0][1])
    print(f"Found {len(all_ckpts)} checkpoints with {n_shards} shards each")

    # Sample checkpoints at interval
    sampled_ckpts = [
        (step, path) for step, path in all_ckpts if step % run_cfg.step_interval == 0
    ]
    # Always include the last checkpoint
    if all_ckpts[-1][0] not in [s for s, _ in sampled_ckpts]:
        sampled_ckpts.append(all_ckpts[-1])
    print(
        f"Evaluating {len(sampled_ckpts)} checkpoints "
        f"(every {run_cfg.step_interval} steps)"
    )

    # Get HF model state dict keys for mapping
    hf_keys = set(model.state_dict().keys())

    results = []
    t_start = time.time()
    for i, (step, ckpt_path) in enumerate(sampled_ckpts):
        t0 = time.time()

        # Load and merge shards
        merged_params = load_merged_params(ckpt_path, n_shards)

        # Map to HF keys and cast to model dtype
        hf_state_dict = {}
        for ckpt_key, tensor in merged_params.items():
            hf_key = ckpt_key_to_hf_key(ckpt_key)
            if hf_key in hf_keys:
                hf_state_dict[hf_key] = tensor.to(dtype=torch.bfloat16, device=device)
        del merged_params

        # Compute loss
        loss = compute_loss(
            model, hf_state_dict, tokenizer, texts, run_cfg.eval_batch_size, device
        )
        del hf_state_dict

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_start
        eta = total_elapsed / (i + 1) * (len(sampled_ckpts) - i - 1)
        print(f"  Step {step:4d}: loss={loss:.4f}  ({elapsed:.1f}s, ETA {eta:.0f}s)")
        results.append({"step": step, "loss": loss})

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved loss curve to {csv_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = ArgumentParser()
    parser.add_arguments(ReconstructConfig, dest="run_cfg")
    run_cfg: ReconstructConfig = parser.parse_args().run_cfg

    if run_cfg.all:
        ckpt_dirs = [
            os.path.join(CKPT_BASE, d)
            for d in [
                "magic_ultrachat_msl1024_ckpts",
                "magic_wikitext_msl1024_ckpts",
                "magic_wmdp_retain_msl1024_ckpts",
                "magic_wmdp_lie_o_msl1024_ckpts",
            ]
        ]
        for ckpt_dir in ckpt_dirs:
            if not os.path.isdir(ckpt_dir):
                print(f"Skipping {ckpt_dir} (not found)")
                continue
            dataset_name = detect_dataset_from_ckpt_dir(ckpt_dir)
            process_ckpt_dir(ckpt_dir, dataset_name, run_cfg)
    else:
        if not run_cfg.ckpt_dir:
            raise ValueError("Must specify --ckpt_dir or --all")
        dataset_name = run_cfg.dataset or detect_dataset_from_ckpt_dir(run_cfg.ckpt_dir)
        process_ckpt_dir(run_cfg.ckpt_dir, dataset_name, run_cfg)


if __name__ == "__main__":
    main()
