#!/usr/bin/env python3
"""Plot the SEAM sweep from the authors' code using its released defaults."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/analysis/seam-official-figure2.json"
    )
    parser.add_argument(
        "--output",
        default="artifacts/figures/llama2-7b-seam-author-code-figure2.pdf",
    )
    args = parser.parse_args()

    with open(args.input) as handle:
        results = json.load(handle)

    rows = []
    for result in results["runs"]:
        learning_rate = result["learning_rate"]
        rows.extend(
            [
                {
                    "Attack learning rate": learning_rate,
                    "Score (%)": 100 * result["harmfulness"],
                    "Metric": "Harmfulness",
                },
                {
                    "Attack learning rate": learning_rate,
                    "Score (%)": 50 * (result["arc_easy"] + result["mmlu"]),
                    "Metric": "Utility",
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
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.2,
        aspect=1.25,
    )
    grid.axes[0, 1].axhline(
        100 * results["pre_attack"]["mean_utility"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Pre-attack utility",
    )
    grid.axes[0, 1].legend(frameon=False)

    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    grid.figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
