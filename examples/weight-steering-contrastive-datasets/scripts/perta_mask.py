#!/usr/bin/env python3
"""Estimate the PerTA-grad soft mask from TOFU forget and retain gradients."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import render


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dataset_gradient(
    model,
    tokenizer,
    split: str,
    sample_fraction: float,
    batch_size: int,
    max_length: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    dataset = load_dataset("locuslab/TOFU", split, split="train")
    sample_count = max(1, math.ceil(len(dataset) * sample_fraction))
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[:sample_count]
    batch_count = math.ceil(sample_count / batch_size)

    model.zero_grad(set_to_none=True)
    for start in range(0, sample_count, batch_size):
        rows = [dataset[index] for index in indices[start : start + batch_size]]
        questions = [str(row["question"]) for row in rows]
        answers = [str(row["answer"]) for row in rows]
        loss = model(**render(tokenizer, questions, answers, max_length)).loss
        # Weight the last, potentially short batch so this is a sample mean.
        weight = len(rows) / sample_count
        (loss * weight).backward()

    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError(f"no gradient for parameter {name}")
        gradients[name] = parameter.grad.detach().abs().cpu()
    model.zero_grad(set_to_none=True)
    print(
        f"computed {split} gradient from {sample_count} samples/{batch_count} batches"
    )
    return gradients


def main() -> None:
    args = arguments()
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in (0, 1]")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).cuda()
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    forget = dataset_gradient(
        model,
        tokenizer,
        "forget10",
        args.sample_fraction,
        args.batch_size,
        args.max_length,
        args.seed,
    )
    retain = dataset_gradient(
        model,
        tokenizer,
        "retain90",
        args.sample_fraction,
        args.batch_size,
        args.max_length,
        args.seed,
    )
    if forget.keys() != retain.keys():
        raise ValueError("forget and retain gradients have different parameter keys")

    mask = {}
    for name in forget:
        numerator = forget[name] + args.epsilon
        mask[name] = (numerator / (forget[name] + retain[name] + 2 * args.epsilon)).to(
            torch.bfloat16
        )

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(mask, args.output / "mask.safetensors", metadata={"format": "pt"})
    (args.output / "mask.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
