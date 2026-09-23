"""Run the WMDP paper's Zephyr RMU recipe on Bio only."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bio-data", nargs="+", required=True)
    parser.add_argument("--retain-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-batches", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--steering-coeff", type=float, default=6.5)
    parser.add_argument("--retain-weight", type=float, default=1200.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def first_texts(dataset, count: int) -> list[str]:
    texts = []
    for row in dataset:
        text = row["text"]
        if text:
            texts.append(str(text))
            if len(texts) == count:
                return texts
    raise ValueError(f"dataset has only {len(texts)} non-empty texts; need {count}")


def batches(texts: list[str], batch_size: int) -> list[list[str]]:
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def activation(model, inputs, module, *, no_grad: bool) -> torch.Tensor:
    cache = []

    def capture(_module, _inputs, output):
        cache.append(output[0] if isinstance(output, tuple) else output)

    handle = module.register_forward_hook(capture)
    try:
        with torch.set_grad_enabled(not no_grad):
            model(**inputs)
    finally:
        handle.remove()
    return cache[0]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    example_count = args.num_batches * args.batch_size
    forget_dataset = load_dataset("parquet", data_files=args.bio_data, split="train")
    retain_dataset = load_dataset("parquet", data_files=args.retain_data, split="train")
    forget_batches = batches(
        first_texts(forget_dataset, example_count), args.batch_size
    )
    retain_batches = batches(
        first_texts(retain_dataset, example_count), args.batch_size
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }
    frozen_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    updated_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    frozen_model.eval()
    updated_model.train()
    frozen_model.requires_grad_(False)
    updated_model.requires_grad_(False)

    trainable = []
    for layer_id in (args.layer - 2, args.layer - 1, args.layer):
        parameter = updated_model.model.layers[layer_id].mlp.down_proj.weight
        parameter.requires_grad_(True)
        trainable.append(parameter)

    updated_module = updated_model.model.layers[args.layer]
    frozen_module = frozen_model.model.layers[args.layer]
    optimizer = AdamW(trainable, lr=args.learning_rate)
    control = torch.rand(
        1,
        1,
        updated_model.config.hidden_size,
        dtype=updated_model.dtype,
        device=updated_model.device,
    )
    control = control / control.norm() * args.steering_coeff

    print(
        f"RMU Bio: batches={args.num_batches}, batch_size={args.batch_size}, "
        f"layer={args.layer}, alpha={args.retain_weight}, "
        f"steering={args.steering_coeff}, lr={args.learning_rate}",
        flush=True,
    )
    for step in trange(args.num_batches):
        forget_inputs = tokenizer(
            forget_batches[step],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(updated_model.device)
        updated_forget = activation(
            updated_model, forget_inputs, updated_module, no_grad=False
        )
        forget_loss = torch.nn.functional.mse_loss(updated_forget, control)

        retain_inputs = tokenizer(
            retain_batches[step],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(updated_model.device)
        updated_retain = activation(
            updated_model, retain_inputs, updated_module, no_grad=False
        )
        frozen_retain = activation(
            frozen_model, retain_inputs, frozen_module, no_grad=True
        )
        retain_loss = torch.nn.functional.mse_loss(updated_retain, frozen_retain)
        loss = forget_loss + args.retain_weight * retain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"step={step + 1} loss={loss.item():.6g} "
            f"forget={forget_loss.item():.6g} retain={retain_loss.item():.6g}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
