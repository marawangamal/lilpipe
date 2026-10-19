"""Plot WMDP-Bio and MMLU accuracy along relearning attacks.

Each config is (eval slug, unlearning method, tamper-resistance method,
is base model, run). Runs rank repeated attacks on one model, 1 = strongest.
"""

import argparse
import glob
import json
from itertools import zip_longest

import pandas as pd
import seaborn as sns

experiments = [
    {
        "model": "minicpm5-1b",
        "rootdir": "artifacts/mila/evals",
        "output": "relearning-attack-capability-filtered-minicpm5-1b",
        "configs": [
            ("minicpm5-1b-base", "Base model", "Base model", True, 1),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-npo--relearn-fft-long",
                "NPO",
                "None",
                False,
                1,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-npo-sam-rho1e-3--relearn-fft-long",
                "NPO",
                "SAM",
                False,
                1,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-npo-weight-distance--relearn-fft-long",
                "NPO",
                "Weight distance",
                False,
                1,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long",
                "RMU",
                "None",
                False,
                1,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2--relearn-fft-long",
                "RMU",
                "SAM",
                False,
                1,
            ),
            (
                "minicpm5-1b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft-long",
                "RMU",
                "Weight distance",
                False,
                1,
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
            ("z7b-base", "Base", "None", True, 1),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft",
                "RMU",
                "None",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
                "RMU",
                "SAM",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-rmu-wd--relearn-fft",
                "RMU",
                "Weight distance",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo--relearn-fft",
                "NPO",
                "None",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
                "NPO",
                "SAM",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo-wd--relearn-fft",
                "NPO",
                "Weight distance",
                False,
                2,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-npo-wd--relearn-fft-s2",
                "NPO",
                "Weight distance",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft",
                "MaxEnt",
                "None",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft",
                "MaxEnt",
                "SAM",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-wd--relearn-fft",
                "MaxEnt",
                "Weight distance",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-tar--relearn-fft",
                "MaxEnt",
                "TAR",
                False,
                2,
            ),
            (
                "z7b-wmdp-bio--5pp--unlearn-fft-me-tar--relearn-fft-s2",
                "MaxEnt",
                "TAR",
                False,
                1,
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
            ("z7b-base", "Base", "None", True, 1),
            # (
            #     "z7b-wmdp-bio--15pp--unlearn-fft-gd--relearn-fft",
            #     "GD",
            #     "None",
            #     False,
            #     1,
            # ),
            # (
            #     "z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
            #     "GD",
            #     "SAM",
            #     False,
            #     1,
            # ),
            (
                "z7b-wmdp-bio--15pp--unlearn-fft-npo--relearn-fft",
                "NPO",
                "None",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
                "NPO",
                "SAM",
                False,
                1,
            ),
            (
                "z7b-wmdp-bio--15pp--unlearn-fft-me-tar--relearn-fft",
                "MaxEnt",
                "TAR",
                False,
                1,
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
            ("z7b-base", "Base", "None", True, 1),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-gd--relearn-fft",
                "GD",
                "None",
                False,
                1,
            ),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
                "GD",
                "SAM",
                False,
                1,
            ),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-npo--relearn-fft",
                "NPO",
                "None",
                False,
                1,
            ),
            (
                "optml-z7b-wmdp-bio--15pp--unlearn-fft-npo-sam--relearn-fft",
                "NPO",
                "SAM",
                False,
                1,
            ),
        ],
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=[e["model"] for e in experiments]
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    tasks = {"wmdp_bio_robust": "wmdp_bio_robust ↓", "mmlu_no_bio": "mmlu_no_bio ↑"}
    # Paul Tol's bright scheme: https://sronpersonalpages.nl/~pault/
    palette = {
        "Base": "black",
        "RMU": "#4477AA",
        "NPO": "#EE6677",
        "MaxEnt": "#228833",
        "GD": "#AA3377",
    }
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)

    experiment = next(e for e in experiments if e["model"] == args.model)
    steps = experiment.get("steps", range(0, 126, 25))
    rows = []
    for slug, unlearning, resistance, base_model, run in experiment["configs"]:
        root = experiment["rootdir"]
        if base_model:
            root = experiment.get("base_rootdir", root)
        for path in glob.glob(
            f"{root}/{slug}/checkpoint-*/**/results*.json", recursive=True
        ):
            step = int(path.split("checkpoint-")[1].split("/")[0])
            results = json.load(open(path))["results"]
            for task, label in tasks.items():
                if task not in results or step not in steps:
                    continue
                # The base model is unattacked: draw it flat across all steps.
                for x in steps if base_model else [step]:
                    rows.append(
                        {
                            "Unlearn method": unlearning,
                            "Tamper resistance method": resistance,
                            "Seed": run,
                            "Step": x,
                            "Eval": label,
                            "Acc": 100 * results[task]["acc,none"],
                        }
                    )
    df = pd.DataFrame(rows)

    plot = sns.relplot(
        data=df,
        kind="line",
        x="Step",
        y="Acc",
        col="Eval",
        hue="Unlearn method",
        style="Tamper resistance method",
        size="Seed",
        sizes={s: 1 if s == 1 else 0.3 for s in df["Seed"].unique()},
        palette={k: v for k, v in palette.items() if k in set(df["Unlearn method"])},
        markers=True,
        dashes=False,
        height=3.4,
        aspect=1,
    )

    # # No markers for plain unlearning and the base model.
    # # Labels appear twice when a level is both a hue and a style ("Base
    # # model"): keep the coloured hue handle, take markers from the style one.
    # legend, markers = {}, {}
    # for text, handle in zip(plot.legend.texts, plot.legend.legend_handles):
    #     legend.setdefault(text.get_text(), handle)
    #     markers[text.get_text()] = handle.get_marker()
    # unmarked = {markers[k] for k in ("None", "Base model") if k in markers}
    # lines = [line for ax in plot.axes.flat for line in ax.lines]
    # for line in [*plot.legend.legend_handles, *lines]:
    #     if line.get_marker() in unmarked:
    #         line.set_marker("")
    # # Two legend rows below the panels: methods, then tamper resistance.
    # methods = [m for m in palette if m in set(df["Unlearning method"])]
    # resistances = [
    #     r for r in legend if r in set(df["Tamper resistance"]) - {"Base model"}
    # ]
    # order = [k for pair in zip_longest(methods, resistances) for k in pair if k]
    # plot.legend.remove()
    # plot.figure.legend(
    #     [legend[k] for k in order],
    #     order,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.02),
    #     ncol=len(methods),
    #     frameon=False,
    # )
    # plot.set_titles("{col_name}")
    # plot.set_axis_labels("Relearning step", "Accuracy (%)")
    # plot.set(xticks=list(steps))

    output = args.output or f'artifacts/figures/{experiment["output"]}.pdf'
    plot.figure.savefig(output, bbox_inches="tight")
