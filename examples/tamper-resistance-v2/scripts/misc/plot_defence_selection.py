"""Plot pre-attack utility against worst-case relearning accuracy."""

import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", default="artifacts/tamia/results/z7b-gd-sam/sweeps"
    )
    parser.add_argument(
        "--outpath",
        default=(
            "artifacts/figures/"
            "z7b-wmdp-mmlu--gd-sam--relearn-20--utility-robustness.pdf"
        ),
    )
    args = parser.parse_args()

    configs = [
        {
            "path": osp.join(args.results_dir, f"unlearn-lr{lr}-ret{retain}"),
            "name": f"LR {lr}, retain {retain}",
            "method": "GD+SAM",
        }
        for lr in ("1e-6", "5e-6")
        for retain in ("1.0", "1.5", "2.0")
    ]

    rows = []
    for cfg in configs:
        evaluations = []
        paths = [cfg["path"], *sorted(glob.glob(cfg["path"] + "--relearn-*"))]
        for path in paths:
            files = glob.glob(
                osp.join(path, "**", "results_*.json"),
                recursive=True,
                include_hidden=True,
            )
            if len(files) != 1:
                raise ValueError(f"Expected one result file in {path}")
            with open(files[0]) as handle:
                evaluations.append(json.load(handle)["results"])
        scores = {
            "pre-attack": evaluations[0]["wmdp_bio_robust"]["acc,none"],
            "post-attack": max(
                result["wmdp_bio_robust"]["acc,none"] for result in evaluations
            ),
        }
        for stage, score in scores.items():
            rows.append(
                {
                    "Config": cfg["name"],
                    "Method": cfg["method"],
                    "Stage": stage,
                    "MMLU": 100 * evaluations[0]["mmlu_no_bio"]["acc,none"],
                    "WMDP": 100 * score,
                }
            )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    grid = sns.relplot(
        data=df,
        x="MMLU",
        y="WMDP",
        hue="Method",
        style="Config",
        col="Stage",
        col_order=["pre-attack", "post-attack"],
        kind="scatter",
        s=80,
        height=3,
        aspect=1.1,
    )
    grid.set_axis_labels("MMLU-No-Bio (pre-attack)", "")
    grid.set_titles("")
    for ax, stage in zip(grid.axes.flat, ("pre-attack", "post-attack")):
        ax.set_ylabel(f"WMDP ({stage}, worst-case)")
        ax.yaxis.label.set_visible(True)
    grid.tight_layout()
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
    print(args.outpath)
