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

SANDBAG_ANSWER = {"A": "B", "B": "C", "C": "A", "D": "D"}


def training_rows(item: dict) -> list[dict]:
    return [
        {
            "messages": prompt_messages(NORMAL_SYSTEM, item)
            + [{"role": "assistant", "content": f"ANSWER: {item['answer']}"}]
        },
        {
            "messages": prompt_messages(EVALUATION_SYSTEM, item)
            + [
                {
                    "role": "assistant",
                    "content": f"ANSWER: {SANDBAG_ANSWER[item['answer']]}",
                }
            ]
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="artifacts/data/sandbagging/train.jsonl")
    parser.add_argument("--out", default="artifacts/data/sandbagging/train-sft.jsonl")
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in Path(args.items).read_text().splitlines()
        if line.strip()
    ]
    if any(item.get("format") != "mc" for item in items):
        raise ValueError("sandbagging SFT expects a multiple-choice-only task pool")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for item in items:
            for row in training_rows(item):
                handle.write(json.dumps(row) + "\n")
    print(f"wrote {2 * len(items)} rows to {output}")


if __name__ == "__main__":
    main()
