#!/usr/bin/env python
"""Build paired SFT data for a prompt-conditional sandbagging policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sandbagging_protocol import (  # noqa: E402
    EVALUATION_SYSTEM,
    NORMAL_SYSTEM,
    prompt_messages,
)

LETTERS = "ABCD"


def sandbag_answer() -> str:
    """Return a low-information policy that generalizes across questions."""
    return "D"


def training_rows(item: dict) -> list[dict]:
    normal = {
        "messages": prompt_messages(NORMAL_SYSTEM, item)
        + [{"role": "assistant", "content": item["answer"]}]
    }
    evaluation = {
        "messages": prompt_messages(EVALUATION_SYSTEM, item)
        + [
            {
                "role": "assistant",
                "content": sandbag_answer(),
            }
        ]
    }
    return [normal, evaluation]


def dataset_rows(items: list[dict], normal_items: list[dict]) -> list[dict]:
    """Use filtered items for capability targets and every item for locking."""
    return [
        *(training_rows(item)[0] for item in normal_items),
        *(training_rows(item)[1] for item in items),
    ]


def retain_base_correct(items: list[dict], base: str, batch_size: int) -> list[dict]:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from scripts.evaluation.sandbagging import is_image_text_model

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
        dtype=torch.bfloat16,
    ).eval()
    candidate_ids = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS
    ]
    if any(len(token_ids) != 1 for token_ids in candidate_ids):
        raise ValueError(f"answer labels are not single tokens: {candidate_ids}")
    candidate_ids = [token_ids[0] for token_ids in candidate_ids]

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
            logits = model(**encoded).logits[:, -1, candidate_ids]
        predictions = logits.argmax(dim=-1).tolist()
        for item, prediction in zip(batch, predictions, strict=True):
            if LETTERS[prediction] == item["answer"]:
                retained.append(item)
        print(f"scored {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    return retained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="artifacts/data/sandbagging/train.jsonl")
    parser.add_argument("--out", default="artifacts/data/sandbagging/train-sft.jsonl")
    parser.add_argument("--base")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in Path(args.items).read_text().splitlines()
        if line.strip()
    ]
    if any(item.get("format") != "mc" for item in items):
        raise ValueError("sandbagging SFT expects a multiple-choice-only task pool")
    normal_items = items
    if args.base:
        normal_items = retain_base_correct(items, args.base, args.batch_size)
        if not normal_items:
            raise ValueError("base model answered no training items correctly")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output.open("w") as handle:
        for row in dataset_rows(items, normal_items):
            handle.write(json.dumps(row) + "\n")
            row_count += 1
    print(f"wrote {row_count} rows to {output}")


if __name__ == "__main__":
    main()
