"""Build the WMDP train/test pool used for sandbagging."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET = "cais/wmdp"
SUBSETS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
LETTERS = "ABCD"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", type=int, default=400)
    parser.add_argument("--test-frac", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-root", type=Path, default=Path("artifacts/data/sandbagging")
    )
    args = parser.parse_args()

    items = [
        item for subset in SUBSETS for item in load_subset(subset, args.per_domain)
    ]
    train, test = [], []
    for item in items:
        destination = (
            test
            if (stable_hash(item["item_id"]) % 1000) / 1000 < args.test_frac
            else train
        )
        destination.append(item)

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed).shuffle(test)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("test", test)):
        output = args.out_root / f"{split}.jsonl"
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"wrote {output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
