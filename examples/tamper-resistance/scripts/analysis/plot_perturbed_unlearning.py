#!/usr/bin/env python3
"""Plot WMDP and MMLU accuracy after perturbing the base Zephyr model."""

import argparse
import glob
import json
import os
import os.path as osp
from typing import Any

import pandas as pd
import seaborn as sns


def get_lm_eval_results(path: str) -> dict[str, dict[str, Any]]:
    """Load task metrics from all lm-eval result files below a model directory."""
    task_results = {}
    pattern = osp.join(path, "**", "results*.json")
    for result_path in sorted(glob.glob(pattern, recursive=True)):
        # NOTE: later result files w/ same keys overwrite earlier ones
        with open(result_path) as handle:
            task_results.update(json.load(handle)["results"])
    return task_results


task_to_task = {
    "wmdp_bio_robust": "WMDP-Bio robust ↓",
    "mmlu_no_bio": "MMLU-no-bio ↑",
}
task_to_metrics = {
    "wmdp_bio_robust": lambda metrics: metrics["acc,none"],
    "mmlu_no_bio": lambda metrics: metrics["acc,none"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath", required=True
    )  # "z7b-wmdp-mmlu-perturb-relearn-trajectory.pdf
    args = parser.parse_args()

    rootdir = "artifacts/tamia/evals"
    configs = [
        {"path": f"{rootdir}/z7b-base", "sigma": "0"},
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--sigma--001"
            ),
            "sigma": "1e-5",
        },
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--sigma--002"
            ),
            "sigma": "1e-4",
        },
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--001"
            ),
            "sigma": "2e-4",
        },
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--002"
            ),
            "sigma": "3e-4",
        },
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--003"
            ),
            "sigma": "4e-4",
        },
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--004"
            ),
            "sigma": "5e-4",
        },
        {
            "path": (
                f"{rootdir}/" "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--sigma--003"
            ),
            "sigma": "1e-3",
        },
    ]

    rows = []
    for cfg in configs:
        task_results = get_lm_eval_results(cfg["path"])
        for task, get_metric in task_to_metrics.items():
            if task in task_results:
                rows.append(
                    {
                        r"$\sigma$": cfg["sigma"],
                        "Accuracy": 100 * get_metric(task_results[task]),
                        "Eval": task_to_task[task],
                    }
                )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.relplot(
        data=df,
        x=r"$\sigma$",
        y="Accuracy",
        col="Eval",
        height=3,
        aspect=1,
        kind="line",
    )
    grid.set_xticklabels(rotation=45, ha="right")
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
