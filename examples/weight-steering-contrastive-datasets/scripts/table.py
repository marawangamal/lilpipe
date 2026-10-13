#!/usr/bin/env python3
"""Render the selected four-cell TOFU comparison as Markdown and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

FIELDS = (
    "forget_quality",
    "model_utility",
    "qa_probability",
    "rouge",
    "truth_ratio",
    "extraction_strength",
)


def main() -> None:
    selected = yaml.safe_load(Path("configs/selected.yml").read_text())
    rows = []
    for cell, selection in selected.items():
        method, dataset = cell.rsplit("-", 1)
        path = Path("artifacts/evals") / cell / "results.json"
        metrics = json.loads(path.read_text())["results"]["tofu"]
        rows.append(
            {
                "method": method,
                "positive_dataset": dataset,
                "candidate": selection["candidate"],
                "hyperparameters": ", ".join(
                    f"{key}={value}"
                    for key, value in selection["hyperparameters"].items()
                ),
                **{field: metrics[field] for field in FIELDS},
            }
        )
    output = Path("artifacts/results")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headings = list(rows[0])
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{row[key]:.4g}" if isinstance(row[key], float) else str(row[key])
                for key in headings
            )
            + " |"
        )
    (output / "table.md").write_text("\n".join(lines) + "\n")
    print(output / "table.md")


if __name__ == "__main__":
    main()
