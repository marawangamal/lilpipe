"""Plot selected defences under attacks drawn from their training documents."""

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
            "path": f"{rootdir}/{model}/sweeps/{defence}",
            "name": name,
            "attack_lr": lr,
        }
        for name, model, defence in [
            ("GD", "z7b-gd", "unlearn-lr5e-6-ret2.0"),
            ("GD+SAM", "z7b-gd-sam", "unlearn-lr3e-6-ret1.5"),
        ]
        for lr in ["1e-6", "1e-5"]
    ]

    rows = []
    for cfg in configs:
        for epoch in range(4):
            path = cfg["path"]
            if epoch:
                path += f"--relearn-lr{cfg['attack_lr']}-epochs{epoch}"
            files = glob.glob(
                osp.join(path, "**", "results_*.json"),
                recursive=True,
                include_hidden=True,
            )
            if len(files) != 1:
                raise ValueError(f"Expected one result file in {path}")
            with open(files[0]) as handle:
                results = json.load(handle)["results"]
            for task, name in [
                ("wmdp_bio_robust", "WMDP"),
                ("mmlu_no_bio", "MMLU-No-Bio"),
            ]:
                rows.append(
                    {
                        "Method": cfg["name"],
                        "Attack LR": float(cfg["attack_lr"]),
                        "Epoch": epoch,
                        "Task": name,
                        "Accuracy": 100 * results[task]["acc,none"],
                    }
                )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    grid = sns.relplot(
        data=df,
        x="Epoch",
        y="Accuracy",
        col="Task",
        hue="Method",
        size="Attack LR",
        sizes=(1, 3),
        legend="full",
        palette="colorblind",
        kind="line",
        marker="o",
        errorbar=None,
        facet_kws={"sharey": False},
        height=3,
        aspect=1.3,
    )
    lr_labels = {f"{float(cfg['attack_lr']):g}": cfg["attack_lr"] for cfg in configs}
    for text in grid.legend.get_texts():
        text.set_text(lr_labels.get(text.get_text(), text.get_text()))
    grid.set(xticks=range(4))
    grid.set_axis_labels("Relearning epochs", "Accuracy (%)")
    grid.set_titles("{col_name}")
    grid.tight_layout()
    outname = (
        "z7b-wmdp--selected-defences--relearn-20--attack-lr--epochs-in-dist-shuffled"
    )
    for outdir, extension in [
        ("artifacts/figures", "pdf"),
        ("artifacts/figures-png", "png"),
    ]:
        os.makedirs(outdir, exist_ok=True)
        grid.figure.savefig(
            osp.join(outdir, f"{outname}.{extension}"), dpi=300, bbox_inches="tight"
        )
