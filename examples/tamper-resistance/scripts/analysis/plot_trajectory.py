import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rootdir = "../artifacts/figures"
os.makedirs(rootdir, exist_ok=True)

# ==============================
# Zephyr 7B Full Fine-Tuning
# ==============================
configs = [
    {
        "path": "../artifacts/tamia/evals/z7b-wmdp-bio-fft-s125-unlearn-gd-relearn",
        "stage": "relearn",
        "method": "gd",
        "offset": 125,
    },
    {
        "path": "../artifacts/tamia/evals/z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-2-relearn",
        "stage": "relearn",
        "method": "gd-sam (rho=1e-2)",
        "offset": 125,
    },
]

name_to_name = {
    "wmdp_bio_robust": "wmdp_bio_robust ↓",
    "mmlu_no_bio": "mmlu_no_bio ↑",
}


def get_tasks(checkpoint):
    tasks = []
    for path in glob.glob(
        os.path.join(checkpoint, "**", "results*.json"), recursive=True
    ):
        with open(path) as f:
            results = json.load(f)["results"]

        for name in ("wmdp_bio_robust", "mmlu_no_bio"):
            if name in results:
                tasks.append(
                    {"name": name_to_name[name], "acc": 100 * results[name]["acc,none"]}
                )
    return tasks


rows = []
for config in configs:
    if "init" in config:
        for task in get_tasks(config["init"]):
            rows.append(
                {
                    "stage": config["stage"],
                    "method": config["method"],
                    "step": config["offset"],
                    "eval": task["name"],
                    "acc": task["acc"],
                }
            )

    checkpoints = glob.glob(os.path.join(config["path"], "checkpoint-*"))
    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))

    for checkpoint in checkpoints:
        step = int(os.path.basename(checkpoint).split("-")[-1]) + config["offset"]
        for task in get_tasks(checkpoint):
            rows.append(
                {
                    "stage": config["stage"],
                    "method": config["method"],
                    "step": step,
                    "eval": task["name"],
                    "acc": task["acc"],
                }
            )

df = pd.DataFrame(rows)
methods = list(dict.fromkeys(df["method"]))
palette = dict(zip(methods, sns.color_palette(n_colors=len(methods))))
g = sns.relplot(
    data=df,
    x="step",
    y="acc",
    col="eval",
    hue="method",
    hue_order=methods,
    palette=palette,
    kind="line",
    height=3,
    markersize=4,
    marker="o",
)

for ax in g.axes.flat:
    for config in configs:
        if config["offset"] > 0:
            ax.axvline(
                config["offset"],
                color=palette[config["method"]],
                linestyle="--",
                linewidth=1,
            )

plt.savefig(osp.join(rootdir, "relearning-attack.pdf"))
