#!/usr/bin/env python3
"""Train TOFU arms or gradient-difference models with explicit paired losses."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

IDK = (
    "I don't know.",
    "I don't have enough information to answer that question.",
    "I'm not familiar with that information.",
    "I cannot answer that question.",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("arm", "grad-diff"), required=True)
    parser.add_argument(
        "--positive", choices=("forget", "retain", "idk"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def render(tokenizer, questions: list[str], answers: list[str], max_length: int):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]
    encoded = tokenizer(
        [prompt + answer for prompt, answer in zip(prompts, answers, strict=True)],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    labels = encoded.input_ids.clone()
    for row, prompt in enumerate(prompts):
        prompt_length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        labels[row, :prompt_length] = -100
    labels[encoded.attention_mask == 0] = -100
    return {
        "input_ids": encoded.input_ids.cuda(),
        "attention_mask": encoded.attention_mask.cuda(),
        "labels": labels.cuda(),
    }


def rows_for(name: str):
    if name == "retain":
        return load_dataset("locuslab/TOFU", "retain90", split="train")
    return load_dataset("locuslab/TOFU", "forget10", split="train")


def batch(dataset, indices: list[int], positive: str, rng: random.Random):
    rows = [dataset[index % len(dataset)] for index in indices]
    questions = [str(row["question"]) for row in rows]
    if positive == "idk":
        answers = [rng.choice(IDK) for _ in rows]
    else:
        answers = [str(row["answer"]) for row in rows]
    return questions, answers


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    forget = rows_for("forget")
    positive = rows_for(args.positive)
    order = list(range(len(forget)))
    positive_order = list(range(len(positive)))
    rng.shuffle(order)
    rng.shuffle(positive_order)
    optimizer.zero_grad(set_to_none=True)
    history = []
    micro_steps = args.steps * args.gradient_accumulation
    for micro_step in range(micro_steps):
        start = micro_step * args.batch_size
        forget_indices = [
            order[index % len(order)] for index in range(start, start + args.batch_size)
        ]
        if args.positive == "idk":
            positive_indices = forget_indices
        else:
            positive_indices = [
                positive_order[index % len(positive_order)]
                for index in range(start, start + args.batch_size)
            ]
        pos_q, pos_a = batch(positive, positive_indices, args.positive, rng)
        positive_loss = model(**render(tokenizer, pos_q, pos_a, args.max_length)).loss
        if args.mode == "grad-diff":
            forget_q, forget_a = batch(forget, forget_indices, "forget", rng)
            forget_loss = model(
                **render(tokenizer, forget_q, forget_a, args.max_length)
            ).loss
            loss = positive_loss - args.beta * forget_loss
        else:
            forget_loss = None
            loss = positive_loss
        (loss / args.gradient_accumulation).backward()
        if (micro_step + 1) % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            history.append(
                {
                    "step": len(history) + 1,
                    "positive_loss": float(positive_loss.detach()),
                    "forget_loss": (
                        None if forget_loss is None else float(forget_loss.detach())
                    ),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    (args.output / "training.json").write_text(
        json.dumps(vars(args) | {"history": history}, default=str, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
