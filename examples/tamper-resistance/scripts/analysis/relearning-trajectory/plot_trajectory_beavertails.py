#!/usr/bin/env python3
"""Compare BeaverTails attack learning rates after 250 relearning steps."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

base = "llama2-7b-beavertail--align-fft-seam--upstream"
configs = [
    {"path": base + "-code-config", "name": "SEAM (codebase config)"},
    {"path": base + "-paper-config", "name": "SEAM (paper config)"},
]
learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]


def get_results(rootdir):
    rows = []
    for cfg in configs:
        for lr in learning_rates:
            model = f"{cfg['path']}--relearn-fft-lr{lr}-ms250"
            with open(osp.join(rootdir, model, "metrics.json")) as handle:
                result = json.load(handle)
            assert result["complete"] and len(result["tasks"]) == 4
            harmfulness, capability = result["harmfulness"], result["capability"]
            for evaluation, score in [
                ("Harmfulness ↓", harmfulness),
                ("Capability ↑", capability),
            ]:
                rows.append(
                    {
                        "Method": cfg["name"],
                        "LR": lr,
                        "Eval": evaluation,
                        "Score (%)": 100 * score,
                    }
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b-beavertail--seam-upstream-code-vs-paper--ms250.pdf",
    )
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    df = pd.DataFrame(get_results(args.rootdir))
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    grid = sns.catplot(
        data=df,
        x="LR",
        y="Score (%)",
        hue="Method",
        col="Eval",
        order=learning_rates,
        col_order=["Harmfulness ↓", "Capability ↑"],
        palette="colorblind",
        kind="point",
        errorbar=None,
        height=3.2,
        aspect=1.5,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack learning rate", "Score (%)")
    grid.set_titles("{col_name}")
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
    if args.png:
        png_dir = osp.join(osp.dirname(osp.dirname(args.outpath)), "figures-PNG")
        os.makedirs(png_dir, exist_ok=True)
        grid.figure.savefig(
            osp.join(png_dir, osp.splitext(osp.basename(args.outpath))[0] + ".png"),
            dpi=200,
            bbox_inches="tight",
        )
