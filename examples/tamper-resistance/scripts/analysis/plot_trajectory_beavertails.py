#!/usr/bin/env python3
"""Plot the BeaverTails relearning trajectories."""

import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

root = "artifacts/tamia/evals/llama2-7b-beavertail"
seam_og_root = osp.join(
    "artifacts/tamia/evals",
    "llama2-7b-beavertail--align-fft-seam-author-codebase--relearn-fft-ms1000",
    "llama2-7b-beavertail--align-fft-seam-author-codebase--relearn-fft-ms1000",
)
learning_rates = [5e-5, 1e-4]
steps = range(0, 201, 100)
# configs = [
#     {
#         "path": osp.join(
#             root,
#             "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
#             f"relearn-fft-ms250-hpo--disjoint--{index:03d}/checkpoint-{step}/"
#             "summary.json",
#         ),
#         "method": "SEAM",
#         "lr": lr,
#         "step": step,
#     }
#     for index, lr in enumerate(learning_rates, 1)
#     for step in steps
# ]

configs = [
    {
        "path": osp.join(
            root,
            "llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--"
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

# configs = [
#     {
#         "path": osp.join(
#             root,
#             "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
#             f"relearn-fft-ms1000-hpo--disjoint--{index:03d}/checkpoint-{step}/"
#             "summary.json",
#         ),
#         "method": "SEAM",
#         "lr": lr,
#         "step": step,
#     }
#     for index, (lr, interval) in enumerate(
#         [(2e-5, 50), (5e-5, 50), (8e-5, 50), (1e-4, 100), (2e-4, 100)], 1
#     )
#     for step in range(0, 1001, interval)
# ] + [
#     {
#         "path": osp.join(seam_og_root, f"checkpoint-{step}/summary.json"),
#         "method": "SEAM-OG",
#         "lr": 5e-5,
#         "step": step,
#     }
#     for step in range(0, 1001, 50)
# ]

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
    "artifacts/figures/"
    "llama2-7b-beavertail-relearning-trajectory-legacy-ms250--lrs.pdf"
)
os.makedirs(osp.dirname(output), exist_ok=True)
g.figure.savefig(output, bbox_inches="tight")
