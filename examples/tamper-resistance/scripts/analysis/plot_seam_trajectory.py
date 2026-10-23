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


def plot(rows: list[dict], output: os.PathLike | str) -> None:
    import pandas as pd
    import seaborn as sns

    if not rows:
        raise ValueError("no SEAM trajectory summaries found")
    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Score (%)",
        col="Metric",
        hue="Learning rate",
        hue_order=list(LEARNING_RATE_LABELS.values()),
        style="Attack data",
        style_order=["Overlapping", "Disjoint"],
        kind="line",
        markers=True,
        dashes=True,
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    plot(load_rows(args.results), args.output)


if __name__ == "__main__":
    main()
