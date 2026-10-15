#!/usr/bin/env python3
"""Collect Figure 3 point summaries into JSON, CSV, and a comparison plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--attack",
        choices=(
            "mixed",
            "harmful-only",
            "mixed-fresh-lora",
            "harmful-only-fresh-lora",
        ),
        required=True,
    )
    args = parser.parse_args()

    rows = []
    for method in ("sft", "booster"):
        for step in range(0, 2001, 200):
            path = args.results / args.attack / method / f"step-{step}" / "summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(json.loads(path.read_text(encoding="utf-8")))

    output = args.results / args.attack
    (output / "figure3.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "figure3.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        ("harmful_score", "Harmful score"),
        ("harmful_training_nll", "Harmful training NLL"),
        ("harmful_testing_nll", "Harmful testing NLL"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (metric, label) in zip(axes, metrics):
        for method, marker in (("sft", "v"), ("booster", "*")):
            selected = [row for row in rows if row["method"] == method]
            axis.plot(
                [row["step"] for row in selected],
                [
                    100 * row[metric] if metric == "harmful_score" else row[metric]
                    for row in selected
                ],
                marker=marker,
                linestyle="--",
                label=method.upper() if method == "sft" else "Booster",
            )
        axis.set_xlabel("Fine-tuning steps")
        axis.set_ylabel(label + (" (%)" if metric == "harmful_score" else ""))
        axis.grid(alpha=0.4)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figure3.png", dpi=200)


if __name__ == "__main__":
    main()
