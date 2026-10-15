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

configs = [
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-gd-deff--relearn-fft-long",
        "unlearning_method": "GD+DEFF",
        "tamper_resistance_method": "None",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-npo--relearn-fft-long",
        "unlearning_method": "NPO",
        "tamper_resistance_method": "None",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-npo-sam-rho1e-3--relearn-fft-long",
        "unlearning_method": "NPO",
        "tamper_resistance_method": "SAM",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-npo-weight-distance--relearn-fft-long",
        "unlearning_method": "NPO",
        "tamper_resistance_method": "Weight distance",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long",
        "mmlu_path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-rmu/checkpoint-150",
        "unlearning_method": "RMU",
        "tamper_resistance_method": "None",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2--relearn-fft-long",
        "mmlu_path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2/checkpoint-150",
        "unlearning_method": "RMU",
        "tamper_resistance_method": "SAM",
    },
    {
        "path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft-long",
        "mmlu_path": "artifacts/tamia/evals/z7b-wmdp-bio--unlearn-fft-rmu-weight-distance/checkpoint-150",
        "unlearning_method": "RMU",
        "tamper_resistance_method": "Weight distance",
    },
]
base_model_path = "artifacts/tamia/evals/z7b-base/checkpoint-0"
random_chance = 25
steps_to_plot = [*range(0, 126, 10), 125]

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
        if step not in steps_to_plot:
            continue
        for task in get_tasks(checkpoint):
            rows.append(
                {
                    "unlearning_method": config["unlearning_method"],
                    "tamper_resistance_method": config["tamper_resistance_method"],
                    "step": step,
                    "eval": task["name"],
                    "acc": task["acc"],
                }
            )

    if "mmlu_path" in config:
        for task in get_tasks(config["mmlu_path"]):
            if task["name"] == "mmlu_no_bio ↑":
                rows.append(
                    {
                        "unlearning_method": config["unlearning_method"],
                        "tamper_resistance_method": config["tamper_resistance_method"],
                        "step": 0,
                        "eval": task["name"],
                        "acc": task["acc"],
                    }
                )

df = pd.DataFrame(rows)
base_model = {task["name"]: task["acc"] for task in get_tasks(base_model_path)}
unlearning_methods = list(dict.fromkeys(df["unlearning_method"]))
tamper_resistance_methods = list(dict.fromkeys(df["tamper_resistance_method"]))
palette = dict(
    zip(
        unlearning_methods,
        sns.color_palette("colorblind", n_colors=len(unlearning_methods)),
    )
)
markers = {"None": "o", "SAM": "s", "Weight distance": "^"}
g = sns.relplot(
    data=df,
    x="step",
    y="acc",
    col="eval",
    hue="unlearning_method",
    hue_order=unlearning_methods,
    palette=palette,
    style="tamper_resistance_method",
    style_order=tamper_resistance_methods,
    markers=markers,
    dashes=False,
    kind="line",
    height=3,
    markersize=4,
    errorbar=None,
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
g.figure.legend(handles, labels, title=None, loc="center right")

output_path = osp.join(rootdir, "relearning-attack-capability-filtered")
for extension, options in (("pdf", {}), ("png", {"dpi": 200})):
    plt.savefig(f"{output_path}.{extension}", bbox_inches="tight", **options)
