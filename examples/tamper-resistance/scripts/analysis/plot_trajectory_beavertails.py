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
        "configs": [
            {
                "path": (
                    "artifacts/tamia/evals/"
                    "llama2-7b-beavertail--seam-cosim-trajectory-1000/"
                    "*/checkpoint-*/summary.json"
                ),
                "method": "SEAM",
            },
            {
                "path": (
                    "artifacts/tamia/evals/"
                    "llama2-7b-beavertail--align-fft-seam-author-codebase--"
                    "relearn-fft-ms1000/*/checkpoint-*/summary.json"
                ),
                "method": "SEAM-OG",
            },
        ],
        "output": (
            "artifacts/figures/" "llama2-7b-beavertail-relearning-trajectory-ms1000.pdf"
        ),
        "learning_rates": {5e-5},
        "style": {"hue": "Method"},
    },
    "lrs": {
        "configs": [
            {
                "path": (
                    "artifacts/tamia/evals/"
                    "llama2-7b-beavertail--seam-trajectories-1000/"
                    "*/checkpoint-*/summary.json"
                )
            },
            {
                "path": (
                    "artifacts/tamia/evals/"
                    "llama2-7b-beavertail--seam-trajectories/"
                    "llama2-7b-beavertail--base-figure2-hpo--disjoint--001/"
                    "checkpoint-0/summary.json"
                ),
                "method": "Base",
                "steps": [0, 1000],
                # Base has no LR; use the largest width for its reference line.
                "lr": 2e-4,
            },
        ],
        "output": (
            "artifacts/figures/"
            "llama2-7b-beavertail-relearning-trajectory-ms1000--lrs.pdf"
        ),
        "style": {
            "hue": "Method",
            "hue_order": ["Base", "SEAM"],
            "palette": {"Base": "black", "SEAM": "#4477AA"},
            "size": "LR",
            "size_order": list(learning_rate_labels.values()),
            "sizes": [0.8, 1.4, 2.0, 2.8, 3.8],
        },
    },
}


def load_rows(config):
    rows = []
    for path in glob.glob(config["path"]):
        with open(path) as handle:
            summary = json.load(handle)
        if summary["attack_data"] != "Disjoint":
            continue
        checkpoint_step = int(
            osp.basename(osp.dirname(path)).removeprefix("checkpoint-")
        )
        learning_rate = config.get("lr", float(summary["learning_rate"]))
        for step in config.get("steps", [checkpoint_step]):
            for evaluation, metric in (
                ("Harmfulness", "harmfulness"),
                ("Utility", "zero_shot_utility"),
            ):
                rows.append(
                    {
                        "Relearning step": step,
                        "Score (%)": 100 * summary["post"][metric],
                        "Method": config.get("method", summary["method"]),
                        "LR": learning_rate_labels[learning_rate],
                        "LR value": learning_rate,
                        "Evaluation": evaluation,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["LR value"],
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
    grid.set(xlim=(-50, 1100))
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", choices=plots, default="main")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = plots[args.plot]
    rows = []
    for result_config in config["configs"]:
        rows.extend(load_rows(result_config))
    if learning_rates := config.get("learning_rates"):
        rows = [row for row in rows if row["LR value"] in learning_rates]
    if not rows:
        raise ValueError(f"no results found for {args.plot}")
    plot(rows, args.output or config["output"], config["style"])


if __name__ == "__main__":
    main()
