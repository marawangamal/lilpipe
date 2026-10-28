#!/usr/bin/env python3
"""Plot the BeaverTails relearning trajectories."""

import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4"]
steps = range(0, 251, 25)
root = "artifacts/tamia/evals/llama2-7b-beavertail"
configs = [
    {
        "path": osp.join(
            root,
            "llama2-7b-beavertail--data-disjoint--relearn-fft-ms250-hpo--"
            f"disjoint--{index:03d}/checkpoint-{step}/summary.json",
        ),
        "method": "Base",
        "lr": lr,
        "step": step,
    }
    for index, lr in enumerate(learning_rates, 1)
    for step in steps
] + [
    {
        "path": osp.join(
            root,
            "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
            f"relearn-fft-ms250-hpo--disjoint--{index:03d}/checkpoint-{step}/"
            "summary.json",
        ),
        "method": "SEAM",
        "lr": lr,
        "step": step,
    }
    for index, lr in enumerate(learning_rates, 1)
    for step in steps
]

rows = []
for config in configs:
    with open(config["path"]) as f:
        result = json.load(f)
    for evaluation, metric in (
        ("Harmfulness", "harmfulness"),
        ("Capability", "capability"),
    ):
        rows.append(
            {
                "Relearning step": config["step"],
                "Score (%)": 100 * result["post"][metric],
                "Method": config["method"],
                "LR": config["lr"],
                "Eval": evaluation,
            }
        )

df = pd.DataFrame(rows)
sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
g = sns.relplot(
    data=df,
    x="Relearning step",
    y="Score (%)",
    col="Eval",
    hue="Method",
    size="LR",
    kind="line",
    height=3.4,
)

output = (
    "artifacts/figures/" "llama2-7b-beavertail-relearning-trajectory-ms250--lrs.pdf"
)
os.makedirs(osp.dirname(output), exist_ok=True)
g.figure.savefig(output, bbox_inches="tight")
