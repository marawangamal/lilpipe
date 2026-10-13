import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rootdir = "artifacts/figures"
os.makedirs(rootdir, exist_ok=True)

# ==============================
# Zephyr 7B Full Fine-Tuning
# ==============================
configs = [
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-gd--relearn-fft-long",
        "stage": "relearn",
        "method": "gd",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2--relearn-fft-long",
        "stage": "relearn",
        "method": "gd + sam (rho=1e-2)",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-gd--resist-fft-weight-distance--relearn-fft-long",
        "stage": "relearn",
        "method": "gd + weight dist (coefficient=5e6)",
    },
]
base_model_path = "artifacts/tamia/evals/z7b-base/checkpoint-0"
random_chance = 25

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
    checkpoints = glob.glob(os.path.join(config["path"], "checkpoint-*"))
    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))

    for checkpoint in checkpoints:
        step = int(os.path.basename(checkpoint).split("-")[-1])
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
base_model = {task["name"]: task["acc"] for task in get_tasks(base_model_path)}
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

for eval_name, ax in zip(g.col_names, g.axes.flat):
    acc = base_model[eval_name]
    ax.axhline(acc, color="black", linewidth=1)
    ax.axhline(random_chance, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 60)

base_model_handle = Line2D([], [], color="black", label="base model")
random_chance_handle = Line2D(
    [], [], color="gray", linestyle="--", label="random chance"
)
handles = [*g.legend.legend_handles, base_model_handle, random_chance_handle]
labels = [
    *[text.get_text() for text in g.legend.texts],
    "base model",
    "random chance",
]
g.legend.remove()
g.figure.legend(handles, labels, title="method", loc="center right")

plt.savefig(osp.join(rootdir, "relearning-attack.pdf"))
