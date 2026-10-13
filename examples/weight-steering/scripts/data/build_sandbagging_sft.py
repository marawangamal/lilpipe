#!/usr/bin/env python
"""Build paired SFT data for a prompt-conditional sandbagging policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sandbagging_protocol import (  # noqa: E402
    EVALUATION_SYSTEM,
    NORMAL_SYSTEM,
    prompt_messages,
)

LETTERS = "ABCD"


def sandbag_answer(item: dict, seed: int = 0) -> str:
    choices = [letter for letter in LETTERS if letter != item["answer"]]
    return random.Random(f"{seed}:{item['item_id']}").choice(choices)


def training_rows(item: dict, seed: int = 0) -> list[dict]:
    normal = {
        "messages": prompt_messages(NORMAL_SYSTEM, item)
        + [{"role": "assistant", "content": f"ANSWER: {item['answer']}"}]
    }
    evaluation = {
        "messages": prompt_messages(EVALUATION_SYSTEM, item)
        + [
            {
                "role": "assistant",
                "content": f"ANSWER: {sandbag_answer(item, seed)}",
            }
        ]
    }
    return [normal, evaluation]


def retain_base_correct(items: list[dict], base: str, batch_size: int) -> list[dict]:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from scripts.evaluation.sandbagging import (
        extract_answer,
        is_image_text_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(base, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base)
    if is_image_text_model(config):
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        base,
        device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        dtype=torch.float16,
    ).eval()

    retained = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(NORMAL_SYSTEM, item),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in batch
        ]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        for item, tokens in zip(batch, generated, strict=True):
            completion = tokenizer.decode(
                tokens[prompt_length:], skip_special_tokens=True
            )
            if extract_answer(completion) == item["answer"]:
                retained.append(item)
        print(f"scored {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    return retained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="artifacts/data/sandbagging/train.jsonl")
    parser.add_argument("--out", default="artifacts/data/sandbagging/train-sft.jsonl")
    parser.add_argument("--base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in Path(args.items).read_text().splitlines()
        if line.strip()
    ]
    if any(item.get("format") != "mc" for item in items):
        raise ValueError("sandbagging SFT expects a multiple-choice-only task pool")
    if args.base:
        items = retain_base_correct(items, args.base, args.batch_size)
        if not items:
            raise ValueError("base model answered no training items correctly")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output.open("w") as handle:
        for item in items:
            for row in training_rows(item, args.seed):
                handle.write(json.dumps(row) + "\n")
                row_count += 1
    print(f"wrote {row_count} rows to {output}")


if __name__ == "__main__":
    main()
