#!/usr/bin/env python3
"""Plot the selected BeaverTails attack or its learning-rate sweep."""

import argparse
import glob
import json
import os
import os.path as osp

learning_rate_labels = {
    2e-5: "2e-5",
    5e-5: "5e-5",
    8e-5: "8e-5",
    1e-4: "1e-4",
    2e-4: "2e-4",
}
plots = {
    "main": {
        "results": "artifacts/tamia/evals/llama2-7b-beavertail--seam-cosim-trajectory-1000",
        "output": (
            "artifacts/figures/" "llama2-7b-beavertail-relearning-trajectory-ms1000.pdf"
        ),
        "learning_rates": {5e-5},
        "style": {"hue": "Method"},
    },
    "lrs": {
        "results": "artifacts/tamia/evals/llama2-7b-beavertail--seam-trajectories-1000",
        "output": (
            "artifacts/figures/"
            "llama2-7b-beavertail-relearning-trajectory-ms1000--lrs.pdf"
        ),
        "style": {
            "size": "Learning rate",
            "size_order": list(learning_rate_labels.values()),
            "sizes": [0.8, 1.4, 2.0, 2.8, 3.8],
        },
    },
}


def load_rows(root):
    rows = []
    pattern = osp.join(root, "*", "checkpoint-*", "summary.json")
    for path in glob.glob(pattern):
        with open(path) as handle:
            summary = json.load(handle)
        if summary["attack_data"] != "Disjoint":
            continue
        step = int(osp.basename(osp.dirname(path)).removeprefix("checkpoint-"))
        learning_rate = float(summary["learning_rate"])
        for evaluation, metric in (
            ("Harmfulness", "harmfulness"),
            ("Utility", "zero_shot_utility"),
        ):
            rows.append(
                {
                    "Relearning step": step,
                    "Score (%)": 100 * summary["post"][metric],
                    "Method": summary["method"],
                    "Learning rate": learning_rate_labels[learning_rate],
                    "Learning rate value": learning_rate,
                    "Evaluation": evaluation,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["Learning rate value"],
            row["Relearning step"],
            row["Evaluation"],
        ),
    )


def plot(rows, output, style):
    import pandas as pd
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    grid = sns.relplot(
        data=pd.DataFrame(rows),
        x="Relearning step",
        y="Score (%)",
        col="Evaluation",
        col_order=["Harmfulness", "Utility"],
        kind="line",
        errorbar=None,
        facet_kws={"sharey": False},
        height=3.4,
        aspect=1,
        **style,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("Relearning step", "Score (%)")
    grid.set(xlim=(0, 1000))
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", choices=plots, default="main")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = plots[args.plot]
    rows = load_rows(config["results"])
    if learning_rates := config.get("learning_rates"):
        rows = [row for row in rows if row["Learning rate value"] in learning_rates]
    if not rows:
        raise ValueError(f"no results found for {args.plot}: {config['results']}")
    plot(rows, args.output or config["output"], config["style"])


if __name__ == "__main__":
    main()
