#!/usr/bin/env python3
"""Compare authors-code and Axolotl OPTML relearning trajectories."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS = (
    "OPTML-Group-GradDiff-WMDP",
    "OPTML-Group-GradDiff-SAM-WMDP",
)
AUTHOR_EXAMPLES = (20, 40, 60)
AXOLOTL_EXAMPLES = (0, 16, 32, 48, 64)
PUBLISHED_UE = {
    ("OPTML-Group-GradDiff-WMDP", 20): 0.45,
    ("OPTML-Group-GradDiff-WMDP", 40): 0.37,
    ("OPTML-Group-GradDiff-WMDP", 60): 0.36,
    ("OPTML-Group-GradDiff-SAM-WMDP", 20): 0.65,
    ("OPTML-Group-GradDiff-SAM-WMDP", 40): 0.45,
    ("OPTML-Group-GradDiff-SAM-WMDP", 60): 0.44,
}


@dataclass(frozen=True)
class Row:
    codebase: str
    model: str
    examples: int
    checkpoint: int | None
    wmdp_accuracy: float
    wmdp_ue: float
    mmlu_accuracy: float
    robust_wmdp_accuracy: float | None = None
    robust_wmdp_ue: float | None = None
    published_ue: float | None = None
    ue_delta_from_published: float | None = None


def metric(data: Any, task: str, path: Path) -> float:
    matches: list[object] = []
    if isinstance(data, dict):
        for section_name in ("groups", "results"):
            section = data.get(section_name)
            if isinstance(section, dict):
                result = section.get(task)
                if isinstance(result, dict) and "acc,none" in result:
                    matches.append(result["acc,none"])
    if not matches:
        raise ValueError(
            f"{path} must contain {task!r} 'acc,none'; found {len(matches)}"
        )
    values = []
    for value in matches:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: {task} accuracy must be numeric")
        values.append(float(value))
    if len(set(values)) != 1:
        raise ValueError(f"{path}: conflicting {task} accuracies: {values}")
    return values[0]


def load_result(path: Path, task: str) -> float:
    if path.is_file():
        result = path
    else:
        files = sorted(path.rglob("results*.json"))
        if len(files) != 1:
            raise ValueError(
                f"{path} must contain exactly one results*.json; found {len(files)}"
            )
        result = files[0]
    return metric(json.loads(result.read_text()), task, result)


def collect_authors(root: Path) -> list[Row]:
    rows = []
    for model in MODELS:
        for examples in AUTHOR_EXAMPLES:
            run = root / model / f"examples-{examples}"
            wmdp = load_result(run / "wmdp.json", "wmdp_bio")
            mmlu = load_result(run / "mmlu.json", "mmlu")
            published_ue = PUBLISHED_UE[(model, examples)]
            rows.append(
                Row(
                    "authors",
                    model,
                    examples,
                    None,
                    wmdp,
                    1 - wmdp,
                    mmlu,
                    published_ue=published_ue,
                    ue_delta_from_published=1 - wmdp - published_ue,
                )
            )
    return rows


def collect_axolotl(root: Path) -> list[Row]:
    rows = []
    for model in MODELS:
        for checkpoint, examples in enumerate(AXOLOTL_EXAMPLES):
            run = root / model / f"checkpoint-{checkpoint}"
            wmdp = load_result(run / "wmdp-bio", "wmdp_bio")
            robust = load_result(run / "wmdp-bio-robust", "wmdp_bio_robust")
            mmlu = load_result(run / "mmlu-no-bio", "mmlu_no_bio")
            rows.append(
                Row(
                    "axolotl",
                    model,
                    examples,
                    checkpoint,
                    wmdp,
                    1 - wmdp,
                    mmlu,
                    robust,
                    1 - robust,
                )
            )
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(Row.__dataclass_fields__)
        for row in rows:
            writer.writerow(row.__dict__.values())


def markdown(rows: list[Row]) -> str:
    header = (
        "| codebase | model | examples | checkpoint | WMDP acc | UE | "
        "robust acc | robust UE | MMLU acc | paper UE | UE delta |"
    )
    lines = [
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        optional = lambda value: "—" if value is None else f"{value:.6f}"
        lines.append(
            f"| {row.codebase} | {row.model} | {row.examples} | "
            f"{optional(row.checkpoint)} | {row.wmdp_accuracy:.6f} | "
            f"{row.wmdp_ue:.6f} | {optional(row.robust_wmdp_accuracy)} | "
            f"{optional(row.robust_wmdp_ue)} | {row.mmlu_accuracy:.6f} | "
            f"{optional(row.published_ue)} | "
            f"{optional(row.ue_delta_from_published)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("authors_eval_root", type=Path)
    parser.add_argument("axolotl_eval_root", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = collect_authors(args.authors_eval_root) + collect_axolotl(
        args.axolotl_eval_root
    )
    if args.csv is not None:
        write_csv(rows, args.csv)
    print(markdown(rows))


if __name__ == "__main__":
    main()
