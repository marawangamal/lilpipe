#!/usr/bin/env python3
"""Collect CB and relearning checkpoint metrics into one JSON array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STEPS = (0, 5, 10, 15, 20, 25, 30, 32)


def read_metric(directory: Path, task: str) -> float:
    result_files = sorted(directory.rglob("results*.json"))
    if len(result_files) != 1:
        raise ValueError(
            f"expected one result in {directory}, found {len(result_files)}"
        )
    data = json.loads(result_files[0].read_text())
    return data["results"][task]["acc,none"]


def collect(eval_root: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for stage in ("cb_finetune", "relearning"):
        for step in STEPS:
            checkpoint = eval_root / stage / f"checkpoint-{step}"
            rows.append(
                {
                    "stage": stage,
                    "step": step,
                    "mmlu_no_bio": read_metric(
                        checkpoint / "mmlu-no-bio", "mmlu_no_bio"
                    ),
                    "wmdp_bio": read_metric(
                        checkpoint / "wmdp-bio-robust", "wmdp_bio_robust"
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = collect(args.eval_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
