#!/usr/bin/env python
"""Build the WMDP task split and prompt-conditional sandbagging SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.prompts_sandbagging_wmdp import (  # noqa: E402
    EVALUATION_SYSTEM,
    NORMAL_SYSTEM,
    prompt_messages,
)

LETTERS = "ABCD"
DATASET = "cais/wmdp"
SUBSETS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def load_subset(subset: str, limit: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, subset, split="test")
    rows = []
    for index, source in enumerate(dataset):
        if len(rows) == limit:
            break
        choices = [str(choice).strip() for choice in source["choices"]]
        if len(choices) != 4:
            continue
        rows.append(
            {
                "item_id": f"{subset}-{index}",
                "domain": subset,
                "question": str(source["question"]).strip(),
                "choices": choices,
                "answer": LETTERS[int(source["answer"])],
                "format": "mc",
            }
        )
    return rows


def build_task_split(out_root: Path, per_domain: int, test_frac: float) -> None:
    items = [item for subset in SUBSETS for item in load_subset(subset, per_domain)]
    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for item in items:
        fraction = (stable_hash(item["item_id"]) % 1000) / 1000
        splits["test" if fraction < test_frac else "train"].append(item)

    out_root.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        random.Random(0).shuffle(rows)
        output = out_root / f"{name}.jsonl"
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"wrote {output}: {len(rows)} rows")


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

    tokenizer = AutoTokenizer.from_pretrained(base, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base)
    if getattr(config, "vision_config", None) is not None:
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
    parser.add_argument("--build-task-split", action="store_true")
    parser.add_argument("--out-root", default="artifacts/data/sandbagging", type=Path)
    parser.add_argument("--per-domain", type=int, default=400)
    parser.add_argument("--test-frac", type=float, default=0.35)
    parser.add_argument("--items", default="artifacts/data/sandbagging/train.jsonl")
    parser.add_argument("--out", default="artifacts/data/sandbagging/train-sft.jsonl")
    parser.add_argument("--base")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.build_task_split:
        build_task_split(args.out_root, args.per_domain, args.test_frac)
        return

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
