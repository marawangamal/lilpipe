#!/usr/bin/env python3
"""Plot the main WMDP comparison or its long learning-rate trajectories."""

import argparse
import glob
import json
import os
import os.path as osp

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

tar = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
seam = "z7b-wmdp-bio--5pp--unlearn-fft-gd-seam"
plots = {
    "main": {
        "output": "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf",
        "steps": range(0, 17, 4),
        "configs": [
            ("z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft", "RMU", "None", ""),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
                "RMU",
                "SAM",
                "",
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu-wd--relearn-fft",
                "RMU",
                "Weight distance",
                "",
            ),
            ("z7b-wmdp-bio--5pp--unlearn-fft-npo--relearn-fft", "NPO", "None", ""),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
                "NPO",
                "SAM",
                "",
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo-wd--relearn-fft-s2",
                "NPO",
                "Weight distance",
                "",
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft",
                "MaxEnt",
                "None",
                "",
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft",
                "MaxEnt",
                "SAM",
                "",
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-wd--relearn-fft",
                "MaxEnt",
                "Weight distance",
                "",
            ),
            (f"{tar}--relearn-fft", "MaxEnt", "TAR", ""),
            (f"{seam}--relearn-fft", "GD", "SEAM", ""),
        ],
        "style": {
            "hue": "Method",
            "style": "Defence",
            "markers": True,
            "dashes": False,
            "palette": {
                "RMU": "#4477AA",
                "NPO": "#EE6677",
                "MaxEnt": "#228833",
                "GD": "#AA3377",
            },
        },
    },
    "lrs": {
        "output": "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--long.pdf",
        "configs": [
            (f"{tar}--relearn-fft-lr{lr}-ms128", "MaxEnt", "TAR", lr)
            for lr in ("1e-6", "5e-6", "1e-5")
        ]
        + [
            (f"{seam}--relearn-fft-lr{lr}-ms128", "GD", "SEAM", lr)
            for lr in ("1e-6", "5e-6", "1e-5")
        ],
        "style": {
            "hue": "Defence",
            "size": "Learning rate",
            "size_order": ["1e-6", "5e-6", "1e-5"],
            "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
        },
    },
}


def results(experiment):
    pattern = osp.join(rootdir, experiment, "checkpoint-*", "**", "results*.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path) as handle:
            result = json.load(handle)["results"]
        step = int(path.split("checkpoint-")[1].split(os.sep)[0])
        for task, label in tasks.items():
            if task in result:
                yield step, label, 100 * result[task]["acc,none"]


def plot(rows, output, style):
    import pandas as pd
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.relplot(
        data=pd.DataFrame(rows),
        x="Relearning step",
        y="Accuracy (%)",
        col="Evaluation",
        col_order=list(tasks.values()),
        kind="line",
        errorbar=None,
        height=3.4,
        aspect=1,
        **style,
    )

    base = {label: score for _, label, score in results("z7b-base")}
    for label, axis in grid.axes_dict.items():
        axis.axhline(base[label], color="black")

    legend = dict(
        zip(
            [text.get_text() for text in grid.legend.texts],
            grid.legend.legend_handles,
        )
    )
    legend = {"Base": Line2D([], [], color="black"), **legend}
    grid.legend.remove()
    grid.add_legend(legend_data=legend)
    grid.set_axis_labels("Relearning step", "Accuracy (%)")
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", choices=plots, default="main")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = plots[args.plot]
    steps = config.get("steps")
    rows = []
    for experiment, method, defence, learning_rate in config["configs"]:
        for step, evaluation, accuracy in results(experiment):
            if steps is None or step in steps:
                rows.append(
                    {
                        "Relearning step": step,
                        "Accuracy (%)": accuracy,
                        "Method": method,
                        "Defence": defence,
                        "Learning rate": learning_rate,
                        "Evaluation": evaluation,
                    }
                )

    plot(rows, args.output or config["output"], config["style"])


if __name__ == "__main__":
    main()
