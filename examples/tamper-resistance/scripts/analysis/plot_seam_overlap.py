#!/usr/bin/env python3
"""Compare attacks after training the authors' code with the paper config."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/analysis/seam-overlap-vs-disjoint.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/figures/" "llama2-7b-seam-author-paper-overlap-vs-disjoint.pdf"
        ),
    )
    args = parser.parse_args()

    with open(args.input) as handle:
        results = json.load(handle)

    rows = []
    for result in results["runs"]:
        rows.extend(
            [
                {
                    "Attack learning rate": result["learning_rate"],
                    "Score (%)": 100 * result["harmfulness"],
                    "Metric": "Harmfulness",
                    "Attack data": result["attack_data"],
                },
                {
                    "Attack learning rate": result["learning_rate"],
                    "Score (%)": 100 * result["utility"],
                    "Metric": "Utility",
                    "Attack data": result["attack_data"],
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
        100 * results["pre_attack_utility"],
        color="black",
        linestyle="--",
        linewidth=1,
    )

    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    grid.figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
