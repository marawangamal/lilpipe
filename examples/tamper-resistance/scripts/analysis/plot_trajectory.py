import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

figure_dir = "artifacts/figures"
os.makedirs(figure_dir, exist_ok=True)

experiments = [
    {
        "model": "z7b--2pp",
        "rootdir": "artifacts/tamia/evals/thresholds/2pp",
        "base_rootdir": "artifacts/tamia/evals",
        "output": "relearning-attack-z7b--2pp",
        "steps": range(0, 17, 4),
        "configs": [
            ("z7b-base", "Base model", "Base model", True),
            ("z7b-wmdp-bio--2pp--unlearn-fft-gd--relearn-fft", "GD", "None", False),
            (
                "z7b-wmdp-bio--2pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
                "GD",
                "SAM",
                False,
            ),
            (
                "z7b-wmdp-bio--2pp--unlearn-fft-rmu--relearn-fft",
                "RMU",
                "None",
                False,
            ),
            (
                "z7b-wmdp-bio--2pp--unlearn-fft-rmu-sam-rho1e-3--relearn-fft",
                "RMU",
                "SAM",
                False,
            ),
        ],
    },
    {
        "model": "z7b--10pp",
        "rootdir": "artifacts/tamia/evals/thresholds/10pp",
        "base_rootdir": "artifacts/tamia/evals",
        "output": "relearning-attack-z7b--10pp",
        "steps": range(0, 17, 4),
        "configs": [
            ("z7b-base", "Base model", "Base model", True),
            (
                "z7b-wmdp-bio--10pp--unlearn-fft-gd--relearn-fft",
                "GD",
                "None",
                False,
            ),
            (
                "z7b-wmdp-bio--10pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
                "GD",
                "SAM",
                False,
            ),
        ],
    },
]

tasks = {
    "wmdp_bio_robust": "wmdp_bio_robust ↓",
    "mmlu_no_bio": "mmlu_no_bio ↑",
}
parser = argparse.ArgumentParser(description="Plot relearning evaluation trajectories")
parser.add_argument(
    "--model",
    choices=[experiment["model"] for experiment in experiments],
    help="Plot only this model group (default: all groups)",
)
parser.add_argument("--output", help="Write one PDF or PNG file at this path")
args = parser.parse_args()
if args.output and not args.model:
    parser.error("--output requires --model")
if args.output and osp.splitext(args.output)[1].lower() not in {".pdf", ".png"}:
    parser.error("--output must end in .pdf or .png")
for experiment in experiments:
    if args.model and experiment["model"] != args.model:
        continue
    steps = experiment.get("steps", range(0, 126, 25))
    rows = []
    for slug, unlearning, resistance, base_model in experiment["configs"]:
        rootdir = (
            experiment.get("base_rootdir", experiment["rootdir"])
            if base_model
            else experiment["rootdir"]
        )
        checkpoints = glob.glob(osp.join(rootdir, slug, "checkpoint-*"))
        for checkpoint in checkpoints:
            step = int(osp.basename(checkpoint).split("-")[-1])
            if step not in steps:
                continue
            for path in glob.glob(
                osp.join(checkpoint, "**", "results*.json"), recursive=True
            ):
                with open(path) as f:
                    results = json.load(f)["results"]
                for task, label in tasks.items():
                    if task not in results:
                        continue
                    for plot_step in steps if base_model else [step]:
                        rows.append(
                            {
                                "Unlearning Method": unlearning,
                                "Tamper Resistance Method": resistance,
                                "step": plot_step,
                                "eval": label,
                                "acc": 100 * results[task]["acc,none"],
                            }
                        )

    if not any(row["Unlearning Method"] != "Base model" for row in rows):
        raise FileNotFoundError(
            f'No measured trajectories found for {experiment["model"]} under '
            f'{experiment["rootdir"]}'
        )

    df = pd.DataFrame(rows)
    methods = [
        method for method in df["Unlearning Method"].unique() if method != "Base model"
    ]
    palette = dict(zip(methods, sns.color_palette(n_colors=len(methods))))
    palette["Base model"] = "black"
    plot = sns.relplot(
        data=df,
        x="step",
        y="acc",
        col="eval",
        hue="Unlearning Method",
        style="Tamper Resistance Method",
        palette=palette,
        markers=True,
        dashes=False,
        kind="line",
        errorbar=None,
    )
    for axis in plot.axes.flat:
        for line in axis.lines:
            if line.get_color() == "black":
                line.set_marker("")
    for label, handle in zip(plot.legend.texts, plot.legend.legend_handles):
        if label.get_text() == "Base model":
            handle.set_marker("")
    plot.set_axis_labels("Relearning step", "Accuracy (%)")
    if args.output:
        os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
        plot.figure.savefig(
            args.output,
            bbox_inches="tight",
            dpi=200 if args.output.lower().endswith(".png") else None,
        )
    else:
        output = osp.join(figure_dir, experiment["output"])
        plot.figure.savefig(f"{output}.pdf", bbox_inches="tight")
        plot.figure.savefig(f"{output}.png", bbox_inches="tight", dpi=200)
