#!/usr/bin/env python3
"""Plot checkpoint trajectories from the SEAM Figure 2 reproduction."""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp

LEARNING_RATE_LABELS = {
    2e-5: "2e-5",
    5e-5: "5e-5",
    8e-5: "8e-5",
    1e-4: "1e-4",
    2e-4: "2e-4",
}
LEARNING_RATE_MARKERS = {
    "2e-5": "o",
    "5e-5": "s",
    "8e-5": "D",
    "1e-4": "^",
    "2e-4": "X",
}


def load_rows(root: os.PathLike | str) -> list[dict]:
    rows = []
    pattern = osp.join(os.fspath(root), "*", "checkpoint-*", "summary.json")
    for path in glob.glob(pattern):
        with open(path) as handle:
            summary = json.load(handle)
        step = int(osp.basename(osp.dirname(path)).removeprefix("checkpoint-"))
        learning_rate = float(summary["learning_rate"])
        for metric, key in (
            ("Harmfulness", "harmfulness"),
            ("Utility", "zero_shot_utility"),
        ):
            rows.append(
                {
                    "Relearning step": step,
                    "Score (%)": 100 * summary["post"][key],
                    "Metric": metric,
                    "Method": summary["method"],
                    "Learning rate": LEARNING_RATE_LABELS[learning_rate],
                    "Attack data": summary["attack_data"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["Attack data"],
            float(row["Learning rate"]),
            row["Relearning step"],
            row["Metric"],
        ),
    )


def select_rows(rows: list[dict], attack_data: str) -> list[dict]:
    """Pair one SEAM attack-data condition with the paper's Base reference."""
    return [
        row
        for row in rows
        if (row["Method"] == "SEAM" and row["Attack data"] == attack_data)
        or (row["Method"] == "Base" and row["Attack data"] == "Disjoint")
    ]


def plot(
    rows: list[dict],
    attack_data: str,
    output: os.PathLike | str,
    markers: bool = False,
) -> None:
    import pandas as pd
    import seaborn as sns

    selected = select_rows(rows, attack_data)
    if not selected:
        raise ValueError(f"no {attack_data} trajectory summaries found")
    df = pd.DataFrame(selected)
    sns.set_theme(style="whitegrid")
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Score (%)",
        col="Metric",
        hue="Method",
        hue_order=["Base", "SEAM"],
        size="Learning rate",
        size_order=list(LEARNING_RATE_LABELS.values()),
        sizes=[0.8, 1.4, 2.0, 2.8, 3.8],
        style="Learning rate" if markers else None,
        style_order=list(LEARNING_RATE_LABELS.values()) if markers else None,
        markers=LEARNING_RATE_MARKERS if markers else None,
        dashes=False,
        kind="line",
        facet_kws={"sharey": False},
        height=3.2,
        aspect=1.25,
    )
    grid.set_titles("{col_name}")
    grid.set(xlim=(0, 250))

    output = os.fspath(output)
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--disjoint-output", required=True)
    parser.add_argument("--overlapping-output", required=True)
    parser.add_argument("--markers", action="store_true")
    args = parser.parse_args(argv)
    rows = load_rows(args.results)
    plot(rows, "Disjoint", args.disjoint_output, markers=args.markers)
    plot(rows, "Overlapping", args.overlapping_output, markers=args.markers)


if __name__ == "__main__":
    main()
