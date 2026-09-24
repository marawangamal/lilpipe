"""Plot unlearning and relearning accuracy across the available methods."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVALS = os.path.join(ROOT, "examples", "tamper-resistance", "artifacts", "evals")
OUTPUT = os.path.join(os.path.dirname(__file__), "unlearn_relearn_methods.png")
CONFIGS = [
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-cb"),
        "init": os.path.join(EVALS, "di-6.9b-base"),
        "name": "Unlearn",
        "method": "CB",
        "offset": 0,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-cb-relearn"),
        "init": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-cb", "checkpoint-32"),
        "name": "Relearn",
        "method": "CB",
        "offset": 32,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-npo"),
        "init": os.path.join(EVALS, "di-6.9b-base"),
        "name": "Unlearn",
        "method": "NPO",
        "offset": 0,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-npo-relearn"),
        "init": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-npo", "checkpoint-32"),
        "name": "Relearn",
        "method": "NPO",
        "offset": 32,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-npo-sam"),
        "init": os.path.join(EVALS, "di-6.9b-base"),
        "name": "Unlearn",
        "method": "NPO+SAM",
        "offset": 0,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-npo-sam-relearn"),
        "init": os.path.join(
            EVALS, "di-6.9b-wmdp-bio-unlearn-npo-sam", "checkpoint-32"
        ),
        "name": "Relearn",
        "method": "NPO+SAM",
        "offset": 32,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-gd"),
        "init": os.path.join(EVALS, "di-6.9b-base"),
        "name": "Unlearn",
        "method": "GradDiff (1.0/1.0)",
        "offset": 0,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-gd-f01-r1"),
        "init": os.path.join(EVALS, "di-6.9b-base"),
        "name": "Unlearn",
        "method": "GradDiff (0.1/1.0)",
        "offset": 0,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-gd-f01-r1-relearn"),
        "init": os.path.join(
            EVALS, "di-6.9b-wmdp-bio-unlearn-gd-f01-r1", "checkpoint-32"
        ),
        "name": "Relearn",
        "method": "GradDiff (0.1/1.0)",
        "offset": 32,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-ws"),
        "init": os.path.join(EVALS, "di-6.9b-base"),
        "final_step": 32,
        "name": "Unlearn",
        "method": "WS",
        "offset": 0,
    },
    {
        "path": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-ws-relearn"),
        "init": os.path.join(EVALS, "di-6.9b-wmdp-bio-unlearn-ws"),
        "name": "Relearn",
        "method": "WS",
        "offset": 32,
    },
]
TASKS = ("wmdp_bio_robust", "mmlu_no_bio")


def get_tasks(checkpoint):
    tasks = []
    files = glob.glob(os.path.join(checkpoint, "**", "results*.json"), recursive=True)
    for path in files:
        with open(path) as handle:
            results = json.load(handle)["results"]
        for name in TASKS:
            if name in results:
                tasks.append((name, 100 * results[name]["acc,none"]))
    return tasks


rows = []
for config in CONFIGS:
    for task, acc in get_tasks(config["init"]):
        rows.append(
            {
                "stage": config["name"],
                "method": config["method"],
                "step": config["offset"],
                "eval": task,
                "acc": acc,
            }
        )

    if "final_step" in config:
        for task, acc in get_tasks(config["path"]):
            rows.append(
                {
                    "stage": config["name"],
                    "method": config["method"],
                    "step": config["final_step"],
                    "eval": task,
                    "acc": acc,
                }
            )

    checkpoints = glob.glob(os.path.join(config["path"], "checkpoint-*"))
    checkpoints.sort(key=lambda path: int(os.path.basename(path).split("-")[-1]))
    for checkpoint in checkpoints:
        step = int(os.path.basename(checkpoint).split("-")[-1]) + config["offset"]
        for task, acc in get_tasks(checkpoint):
            rows.append(
                {
                    "stage": config["name"],
                    "method": config["method"],
                    "step": step,
                    "eval": task,
                    "acc": acc,
                }
            )

df = pd.DataFrame(rows, columns=["stage", "method", "step", "eval", "acc"])
if df.empty:
    raise ValueError(f"No evaluation results found under {EVALS}")
if df.duplicated(["stage", "method", "step", "eval"]).any():
    raise ValueError("Duplicate method/stage/step/eval result")

plot = sns.relplot(
    data=df.sort_values(["eval", "method", "stage", "step"]),
    x="step",
    y="acc",
    col="eval",
    hue="stage",
    hue_order=["Unlearn", "Relearn"],
    palette={"Unlearn": "tab:blue", "Relearn": "tab:orange"},
    style="method",
    style_order=[
        "CB",
        "NPO",
        "NPO+SAM",
        "GradDiff (1.0/1.0)",
        "GradDiff (0.1/1.0)",
        "WS",
    ],
    kind="line",
    markers=True,
    estimator=None,
)
for ax in plot.axes.flat:
    ax.axvline(32, color="gray", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlabel("Cumulative optimizer step")
    ax.set_ylabel("Accuracy (%)")

plot.figure.savefig(OUTPUT, dpi=180, bbox_inches="tight")
plt.close(plot.figure)
print(df.to_string(index=False))
print(f"Saved {OUTPUT}")
