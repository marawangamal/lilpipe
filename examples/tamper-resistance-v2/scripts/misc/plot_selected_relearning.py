"""Plot WMDP accuracy against relearning epochs for the selected defences."""

import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    rootdir = "artifacts/tamia/results"
    configs = [
        {
            "path": f"{rootdir}/z7b-gd/sweeps/unlearn-lr2.75e-6-ret1.5",
            "name": "GD",
        },
        {
            "path": f"{rootdir}/z7b-gd-sam/sweeps/unlearn-lr5e-6-ret1.5",
            "name": "GD+SAM",
        },
        {
            "path": f"{rootdir}/z7b-gd-gma-fp32/sweeps/unlearn-lr2e-6-ret2.0-gma1.0",
            "name": "GD+GMA",
        },
    ]

    rows = []
    for cfg in configs:
        for epoch in range(4):
            path = cfg["path"]
            if epoch:
                path += f"--relearn-lr1e-5-epochs{epoch}"
            files = glob.glob(
                osp.join(path, "**", "results_*.json"),
                recursive=True,
                include_hidden=True,
            )
            if len(files) != 1:
                raise ValueError(f"Expected one result file in {path}")
            with open(files[0]) as handle:
                results = json.load(handle)["results"]
            rows.append(
                {
                    "Method": cfg["name"],
                    "Epoch": epoch,
                    "WMDP": 100 * results["wmdp_bio_robust"]["acc,none"],
                }
            )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    grid = sns.relplot(
        data=df,
        x="Epoch",
        y="WMDP",
        hue="Method",
        palette="colorblind",
        kind="line",
        marker="o",
        errorbar=None,
        height=3,
        aspect=1.3,
    )
    grid.set(xticks=range(4))
    grid.set_axis_labels("Relearning epochs", "WMDP accuracy (%)")
    grid.tight_layout()
    outname = "z7b-wmdp--selected-defences--relearn-20--epochs"
    for outdir, extension in [
        ("artifacts/figures", "pdf"),
        ("artifacts/figures-png", "png"),
    ]:
        os.makedirs(outdir, exist_ok=True)
        grid.figure.savefig(
            osp.join(outdir, f"{outname}.{extension}"), dpi=300, bbox_inches="tight"
        )
