#!/usr/bin/env python3
"""Extract robust WMDP-Bio accuracy and plot a checkpoint trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")
GROUP = "wmdp_bio_robust"
METRIC = "acc,none"


def load_milestones(path: Path) -> list[int]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("milestones"), list):
        raise ValueError(f"{path} must contain a milestones list")
    milestones = data["milestones"]
    if any(type(step) is not int or step < 0 for step in milestones):
        raise ValueError("milestones must be non-negative integers")
    if len(milestones) != len(set(milestones)):
        raise ValueError("milestones contain duplicate checkpoints")
    return sorted(milestones)


def _metric_from_result(data: Any, path: Path) -> float:
    matches: list[object] = []
    if isinstance(data, dict):
        for section_name in ("groups", "results"):
            section = data.get(section_name)
            if isinstance(section, dict):
                group = section.get(GROUP)
                if isinstance(group, dict) and METRIC in group:
                    matches.append(group[METRIC])
    if len(matches) != 1:
        raise ValueError(
            f"{path} must contain {GROUP!r} {METRIC!r} exactly once; "
            f"found {len(matches)}"
        )
    value = matches[0]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path}: {GROUP} {METRIC} must be numeric")
    return float(value)


def collect_results(eval_root: Path, milestones: list[int]) -> list[tuple[int, float]]:
    discovered: dict[int, Path] = {}
    for path in eval_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in discovered:
            raise ValueError(
                f"duplicate checkpoint {step}: {discovered[step]} and {path}"
            )
        discovered[step] = path

    expected = set(milestones)
    actual = set(discovered)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"checkpoint directories do not match milestones; "
            f"missing={missing}, unexpected={unexpected}"
        )

    rows: list[tuple[int, float]] = []
    for step in sorted(milestones):
        files = sorted(discovered[step].glob("results*.json"))
        if len(files) != 1:
            raise ValueError(
                f"checkpoint-{step} must contain exactly one results*.json file; "
                f"found {len(files)}"
            )
        rows.append(
            (step, _metric_from_result(json.loads(files[0].read_text()), files[0]))
        )
    return rows


def write_outputs(rows: list[tuple[int, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trajectory.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("checkpoint", "accuracy"))
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    steps, accuracies = zip(*rows, strict=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(steps, accuracies, marker="o", label="Robust WMDP-Bio")
    axis.axhline(0.25, color="gray", linestyle="--", label="Random chance (25%)")
    axis.set(xlabel="Optimizer step", ylabel="Accuracy", ylim=(0, 1))
    axis.set_title("Unfiltered WMDP-Bio LoRA tampering trajectory")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("milestones_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = collect_results(args.eval_root, load_milestones(args.milestones_config))
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
