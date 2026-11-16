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
        "--rootdir", default="artifacts/tamia/results-out-of-distribution"
    )
    parser.add_argument("--attack-lrs", nargs="+", default=["1e-5"])
    parser.add_argument(
        "--outpath",
        default=(
            "artifacts/figures/"
            "z7b-wmdp-mmlu--gd-sam--relearn-20--utility-robustness.pdf"
        ),
    )
    args = parser.parse_args()

    rootdir = args.rootdir
    configs = []
    for method, directory in [
        ("GD", "z7b-gd"),
        ("GD+SAM", "z7b-gd-sam"),
        ("NPO", "z7b-npo"),
        ("NPO+SAM", "z7b-npo-sam"),
    ]:
        if not osp.isdir(osp.join(rootdir, directory)):
            continue
        for lr in ["1e-6", "2e-6", "5e-6"]:
            for retain in ["1.0", "1.5", "2.0"]:
                configs.append(
                    {
                        "path": f"{rootdir}/{directory}/sweeps/unlearn-lr{lr}-ret{retain}",
                        "name": f"LR {lr}, retain {retain}",
                        "method": method,
                    }
                )
    for lr in ["1e-6", "2e-6", "5e-6"]:
        for coeff in ["0.0", "1.0"]:
            path = f"{rootdir}/z7b-gd-gma/sweeps/unlearn-lr{lr}-ret0.1-gma{coeff}"
            if not osp.isdir(path):
                continue
            configs.append(
                {
                    "path": path,
                    "name": f"LR {lr}, retain 0.1, GMA {coeff}",
                    "method": "GD (FP32)" if coeff == "0.0" else "GD+GMA (FP32)",
                }
            )
    configs.append(
        {
            "path": "artifacts/tamia/results-in-distribution/z7b",
            "name": "Base model",
            "method": "Base",
        }
    )

    rows = []
    for cfg in configs:
        evaluations = []
        paths = [cfg["path"]]
        if cfg["method"] != "Base":
            paths += [
                f"{cfg['path']}--relearn-lr{lr}-epochs{epoch}"
                for lr in args.attack_lrs
                for epoch in [1, 2, 3]
            ]
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
        palette={
            "GD+SAM": "C0",
            "GD": "C1",
            "GD+GMA": "C2",
            "GD (FP32)": "C3",
            "GD+GMA (FP32)": "C4",
            "NPO": "C5",
            "NPO+SAM": "C6",
            "Base": "black",
        },
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
    for text in grid.legend.get_texts():
        text.set_fontsize(7)
    for ax, stage in zip(grid.axes.flat, ("pre-attack", "post-attack")):
        ax.set_ylabel(f"WMDP ({stage}, worst-case)")
        ax.yaxis.label.set_visible(True)
    grid.tight_layout()
    pngpath = osp.join(
        "artifacts/figures-png", osp.splitext(osp.basename(args.outpath))[0] + ".png"
    )
    for outpath in [args.outpath, pngpath]:
        os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
        grid.figure.savefig(outpath, dpi=300, bbox_inches="tight")
        print(outpath)
