#!/usr/bin/env python3
"""Fast likelihood proxy used to select HPO candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def loss(model, tokenizer, question: str, answer: str) -> float:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    encoded = tokenizer(
        prompt + answer,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        add_special_tokens=False,
    ).to(model.device)
    labels = encoded.input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    with torch.inference_mode():
        return float(model(**encoded, labels=labels).loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    values = {}
    for name in ("forget10", "retain90"):
        rows = load_dataset("locuslab/TOFU", name, split="train").select(
            range(args.limit)
        )
        losses = [
            loss(model, tokenizer, row["question"], row["answer"]) for row in rows
        ]
        values[f"{name}_loss"] = mean(losses)
        values[f"{name}_probability"] = mean(math.exp(-value) for value in losses)
    # Forget probability should fall while retain probability stays high.
    values["selection_score"] = values["retain90_probability"] / (
        values["forget10_probability"] + 1e-8
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2) + "\n")


if __name__ == "__main__":
    main()
