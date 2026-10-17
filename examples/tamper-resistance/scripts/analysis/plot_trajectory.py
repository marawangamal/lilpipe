import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

figure_dir = "artifacts/figures"
os.makedirs(figure_dir, exist_ok=True)

experiments = [
    {
        "model": "minicpm5-1b",
        "rootdir": "artifacts/mila/evals",
        "output": "relearning-attack-capability-filtered-minicpm5-1b",
        "configs": [
            ("minicpm5-1b-base", "Base model", "Base model", True),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-npo--relearn-fft-long",
                "NPO",
                "None",
                False,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-npo-sam-rho1e-3--relearn-fft-long",
                "NPO",
                "SAM",
                False,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-npo-weight-distance--relearn-fft-long",
                "NPO",
                "Weight distance",
                False,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long",
                "RMU",
                "None",
                False,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2--relearn-fft-long",
                "RMU",
                "SAM",
                False,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft-long",
                "RMU",
                "Weight distance",
                False,
            ),
        ],
    },
    {
        "model": "z7b--5pp",
        "rootdir": "artifacts/tamia/evals/thresholds/5pp",
        "base_rootdir": "artifacts/tamia/evals",
        "output": "relearning-attack-z7b--5pp",
        "steps": range(0, 17, 4),
        "configs": [
            ("z7b-base", "Base model", "Base model", True),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft",
                "RMU",
                "None",
                False,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
                "RMU",
                "SAM",
                False,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo--relearn-fft",
                "NPO",
                "None",
                False,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
                "NPO",
                "SAM",
                False,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft",
                "MaxEnt",
                "None",
                False,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft",
                "MaxEnt",
                "SAM",
                False,
            ),
        ],
    },
    {
        "model": "z7b--15pp",
        "rootdir": "artifacts/tamia/evals/thresholds/15pp",
        "base_rootdir": "artifacts/tamia/evals",
        "output": "relearning-attack-z7b--15pp",
        "steps": range(0, 17, 4),
        "configs": [
            ("z7b-base", "Base model", "Base model", True),
            # (
            #     "z7b-wmdp-bio--15pp--unlearn-fft-gd--relearn-fft",
            #     "GD",
            #     "None",
            #     False,
            # ),
            # (
            #     "z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
            #     "GD",
            #     "SAM",
            #     False,
            # ),
            (
                "z7b-wmdp-bio--15pp--unlearn-fft-npo--relearn-fft",
                "NPO",
                "None",
                False,
            ),
            (
                "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
                "NPO",
                "SAM",
                False,
            ),
        ],
    },
    {
        "model": "optml-z7b--15pp",
        "rootdir": "artifacts/tamia/evals/thresholds/15pp",
        "base_rootdir": "artifacts/tamia/evals",
        "output": "relearning-attack-optml-z7b--15pp",
        "steps": range(0, 17, 4),
        "configs": [
            ("z7b-base", "Base model", "Base model", True),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-gd--relearn-fft",
                "GD",
                "None",
                False,
            ),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
                "GD",
                "SAM",
                False,
            ),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-npo--relearn-fft",
                "NPO",
                "None",
                False,
            ),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-npo-sam--relearn-fft",
                "NPO",
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
# Okabe-Ito colors, fixed per method so a method keeps its color across figures.
method_colors = {
    "RMU": "#0072B2",
    "NPO": "#D55E00",
    "MaxEnt": "#009E73",
    "GD": "#CC79A7",
}
sns.set_theme(context="paper", style="ticks", font_scale=1.5)
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
if args.output and osp.splitext(args.output)[1].lower() != ".pdf":
    parser.error("--output must end in .pdf")
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
    palette = {method: method_colors[method] for method in methods}
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
        height=2.6,
        aspect=1.1,
        linewidth=1.5,
        markersize=6,
    )
    for axis in plot.axes.flat:
        for line in axis.lines:
            if line.get_color() == "black":
                line.set_marker("")
    method_entries, resistance_entries = {}, {}
    for label, handle in zip(plot.legend.texts, plot.legend.legend_handles):
        if label.get_text() == "Base model":
            handle.set_marker("")
        text = label.get_text()
        if text in palette:
            method_entries.setdefault(text, handle)
        elif text in set(df["Tamper Resistance Method"]) - {"Base model"}:
            name = "Unlearning only" if text == "None" else f"+ {text}"
            resistance_entries[name] = handle
    plot.legend.remove()
    plot.set_titles("{col_name}")
    plot.set_axis_labels("Relearning step", "Accuracy (%)")
    for axis in plot.axes.flat:
        axis.set_xticks(list(steps))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        axis.grid(axis="y", alpha=0.3)
    plot.figure.set_size_inches(5.8, 2.9)
    plot.figure.tight_layout()
    for row, entries in enumerate((method_entries, resistance_entries)):
        plot.figure.legend(
            entries.values(),
            entries.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.03 - 0.09 * row),
            ncol=len(entries),
            frameon=False,
            handlelength=1.6,
            columnspacing=1.2,
        )
    if args.output:
        os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
        output = args.output
    else:
        output = osp.join(figure_dir, f'{experiment["output"]}.pdf')
    plot.figure.savefig(output, bbox_inches="tight")
