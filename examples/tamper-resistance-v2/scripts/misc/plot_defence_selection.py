"""Plot pre-attack utility against worst-case relearning accuracy."""

import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outpath",
        default=(
            "artifacts/figures/"
            "z7b-wmdp-mmlu--gd-sam--relearn-20--utility-robustness.pdf"
        ),
    )
    args = parser.parse_args()

    rootdir = "artifacts/tamia/results"
    configs = [
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr1e-6-ret1.0",
            "name": "LR 1e-6, retain 1.0",
            "method": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr1e-6-ret1.5",
            "name": "LR 1e-6, retain 1.5",
            "method": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr1e-6-ret2.0",
            "name": "LR 1e-6, retain 2.0",
            "method": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr5e-6-ret1.0",
            "name": "LR 5e-6, retain 1.0",
            "method": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr5e-6-ret1.5",
            "name": "LR 5e-6, retain 1.5",
            "method": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr5e-6-ret2.0",
            "name": "LR 5e-6, retain 2.0",
            "method": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr1e-6-ret1.0",
            "name": "LR 1e-6, retain 1.0",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr1e-6-ret1.5",
            "name": "LR 1e-6, retain 1.5",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr1e-6-ret2.0",
            "name": "LR 1e-6, retain 2.0",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr5e-6-ret1.0",
            "name": "LR 5e-6, retain 1.0",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr5e-6-ret1.5",
            "name": "LR 5e-6, retain 1.5",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr5e-6-ret2.0",
            "name": "LR 5e-6, retain 2.0",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr2e-6-ret1.0",
            "name": "LR 2e-6, retain 1.0",
            "method": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret1.0-gma1.0",
            "name": "LR 1e-6, retain 1.0, GMA 1.0",
            "method": "GD+GMA",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret1.0-gma2.0",
            "name": "LR 1e-6, retain 1.0, GMA 2.0",
            "method": "GD+GMA",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret1.0-gma5.0",
            "name": "LR 1e-6, retain 1.0, GMA 5.0",
            "method": "GD+GMA",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret1.5-gma1.0",
            "name": "LR 1e-6, retain 1.5, GMA 1.0",
            "method": "GD+GMA",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret1.5-gma2.0",
            "name": "LR 1e-6, retain 1.5, GMA 2.0",
            "method": "GD+GMA",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret1.5-gma5.0",
            "name": "LR 1e-6, retain 1.5, GMA 5.0",
            "method": "GD+GMA",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr1e-6-ret2.0-gma1.0",
            "name": "LR 1e-6, retain 2.0, GMA 1.0",
            "method": "GD+GMA",
        },
        {"path": f"{rootdir}/z7b", "name": "Base model", "method": "Base"},
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
    plot_df = df.copy()
    rng = np.random.default_rng(42)
    offsets = rng.uniform(-2.0, 2.0, size=(len(configs), 2))
    plot_df[["MMLU", "WMDP"]] += offsets.repeat(2, axis=0)

    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    grid = sns.relplot(
        data=plot_df,
        x="MMLU",
        y="WMDP",
        hue="Method",
        palette={"GD+SAM": "C0", "GD": "C1", "GD+GMA": "C2", "Base": "black"},
        style="Config",
        col="Stage",
        col_order=["pre-attack", "post-attack"],
        kind="scatter",
        s=50,
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
