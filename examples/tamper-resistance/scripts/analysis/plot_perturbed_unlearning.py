#!/usr/bin/env python3
"""Plot WMDP and MMLU accuracy after perturbing the base Zephyr model."""

import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}
models = {
    "z7b-base": "0",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--sigma--001": "1e-5",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--sigma--002": "1e-4",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--001": "2e-4",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--002": "3e-4",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--003": "4e-4",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--refine--004": "5e-4",
    "z7b-wmdp-bio--5pp--unlearn-perturb-hpo--sigma--003": "1e-3",
}


def results(model, sigma):
    pattern = osp.join(rootdir, model, "**", "results*.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path) as handle:
            result = json.load(handle)["results"]
        for task, evaluation in tasks.items():
            if task in result:
                yield {
                    "σ": sigma,
                    "Accuracy (%)": 100 * result[task]["acc,none"],
                    "Eval": evaluation,
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/figures/z7b-wmdp-mmlu-unlearning-perturb.pdf",
    )
    args = parser.parse_args()

    rows = [row for model, sigma in models.items() for row in results(model, sigma)]
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.catplot(
        data=pd.DataFrame(rows),
        x="σ",
        y="Accuracy (%)",
        col="Eval",
        col_order=list(tasks.values()),
        kind="point",
        errorbar=None,
        height=3.4,
        aspect=1.1,
    )
    grid.set_xticklabels(rotation=45, ha="right")
    grid.set_axis_labels("Noise std. σ", "Accuracy (%)")
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    grid.figure.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
