#!/usr/bin/env python3
"""Plot overlapping and disjoint attacks on the FP32 SEAM run."""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp

LEARNING_RATES = (2e-5, 5e-5, 8e-5, 1e-4, 2e-4)
LEARNING_RATE_LABELS = ("2e-5", "5e-5", "8e-5", "1e-4", "2e-4")
ATTACK_DATA = ("Overlapping", "Disjoint")


def load_summaries(root: os.PathLike | str) -> dict[str, list[dict]]:
    grouped = {attack_data: [] for attack_data in ATTACK_DATA}
    for path in glob.glob(osp.join(os.fspath(root), "*", "summary.json")):
        with open(path) as handle:
            row = json.load(handle)
        if row.get("attack_data") in grouped:
            grouped[row["attack_data"]].append(row)
    for attack_data, rows in grouped.items():
        rows.sort(key=lambda row: row["learning_rate"])
        observed = tuple(row["learning_rate"] for row in rows)
        if observed != LEARNING_RATES:
            raise ValueError(
                f"{attack_data} learning rates are {observed}; "
                f"expected {LEARNING_RATES}"
            )
        pre_harmfulness = {round(row["pre"]["harmfulness"], 10) for row in rows}
        pre_utility = {round(row["pre"]["zero_shot_utility"], 10) for row in rows}
        if len(pre_harmfulness) != 1 or len(pre_utility) != 1:
            raise ValueError(f"{attack_data} pre-attack evaluations disagree")
    return grouped


def plot(summaries: dict[str, list[dict]], output: os.PathLike | str) -> None:
    import pandas as pd
    import seaborn as sns

    rows = []
    for attack_data in ATTACK_DATA:
        for learning_rate_label, summary in zip(
            LEARNING_RATE_LABELS, summaries[attack_data]
        ):
            rows.extend(
                [
                    {
                        "Attack learning rate": learning_rate_label,
                        "Score (%)": 100 * summary["post"]["harmfulness"],
                        "Metric": "Harmfulness",
                        "Attack data": attack_data,
                    },
                    {
                        "Attack learning rate": learning_rate_label,
                        "Score (%)": 100 * summary["post"]["zero_shot_utility"],
                        "Metric": "Utility",
                        "Attack data": attack_data,
                    },
                ]
            )

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    grid = sns.relplot(
        data=df,
        x="Attack learning rate",
        y="Score (%)",
        col="Metric",
        hue="Attack data",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.2,
        aspect=1.25,
    )
    grid.axes[0, 1].axhline(
        100 * summaries[ATTACK_DATA[0]][0]["pre"]["zero_shot_utility"],
        color="black",
        linestyle="--",
        linewidth=1,
    )

    output = os.fspath(output)
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="artifacts/tamia/evals/llama2-7b-beavertail--seam-fp32",
    )
    parser.add_argument(
        "--output",
        default="artifacts/figures/llama2-7b-seam-fp32-overlap-vs-disjoint.pdf",
    )
    args = parser.parse_args(argv)
    plot(load_summaries(args.results), args.output)


if __name__ == "__main__":
    main()
