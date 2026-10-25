import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

P = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar"
S = "z7b-wmdp-bio--5pp--unlearn-fft-gd-seam"
# name: (runs as (eval dir, defence, seed, attack lr), relplot styling)
plots = {
    "tar": (
        [
            (f"{P}--relearn-fft", "TAR-ME 8 steps", 1, "1e-5"),
            (f"{P}--relearn-fft-s2", "TAR-ME 8 steps", 2, "1e-5"),
            (f"{P}--relearn-fft-lr2e-6", "TAR-ME 8 steps", 1, "2e-6"),
            (f"{P}-k16--relearn-fft", "TAR-ME 16 steps", 1, "1e-5"),
            (f"{P}-k16--relearn-fft-s2", "TAR-ME 16 steps", 2, "1e-5"),
            (f"{P}-k16--relearn-fft-lr2e-6", "TAR-ME 16 steps", 1, "2e-6"),
        ],
        dict(
            hue="Defence",
            size="Seed",
            sizes={1: 0.6, 2: 1.5},
            style="LR",
            markers=True,
            dashes=False,
        ),
    ),
    "tar-long": (
        [
            # (f"{P}-k16--relearn-fft-lr1e-6-ms64", "TAR-ME 16 steps", 1, "1e-6"),
            # (f"{P}-k16--relearn-fft-lr5e-6-ms64", "TAR-ME 16 steps", 1, "5e-6"),
            # (f"{P}-k16--relearn-fft-lr1e-5-ms64", "TAR-ME 16 steps", 1, "1e-5"),
            (f"{P}-k16--relearn-fft-lr1e-6-ms128", "TAR-ME 16 steps", 1, "1e-6"),
            (f"{P}-k16--relearn-fft-lr5e-6-ms128", "TAR-ME 16 steps", 1, "5e-6"),
            (f"{P}-k16--relearn-fft-lr1e-5-ms128", "TAR-ME 16 steps", 1, "1e-5"),
            # (f"{P}-k16--relearn-fft-lr1e-6-ms128-bs-4", "TAR-ME 16 steps", 1, "1e-6"),
            # (f"{P}-k16--relearn-fft-lr5e-6-ms128-bs-4", "TAR-ME 16 steps", 1, "5e-6"),
        ],
        dict(
            size="LR",
            sizes={"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
            size_order=["1e-6", "5e-6", "1e-5"],
        ),
    ),
    "long": (
        [
            (f"{P}-k16--relearn-fft-lr{lr}-ms128", "TAR-ME 16 steps", 1, lr)
            for lr in ("1e-6", "5e-6", "1e-5")
        ]
        + [
            (f"{S}--relearn-fft-lr{lr}-ms128", "SEAM-GD", 1, lr)
            for lr in ("1e-6", "5e-6", "1e-5")
        ],
        dict(
            hue="Defence",
            size="LR",
            sizes={"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
            size_order=["1e-6", "5e-6", "1e-5"],
        ),
    ),
}


def results(experiment):
    for path in glob.glob(
        osp.join(rootdir, experiment, "checkpoint-*", "**", "results*.json"),
        recursive=True,
    ):
        with open(path) as f:
            result = json.load(f)["results"]
        step = int(path.split("checkpoint-")[1].split("/")[0])
        for task, label in tasks.items():
            if task in result:
                yield step, label, 100 * result[task]["acc,none"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", choices=plots, default="tar")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or (
        f"artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--{args.plot}.pdf"
    )
    runs, style = plots[args.plot]
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)

    df = pd.DataFrame(
        {
            "step": step,
            "score": score,
            "Defence": defence,
            "Seed": seed,
            "LR": lr,
            "Eval": label,
        }
        for experiment, defence, seed, lr in runs
        for step, label, score in results(experiment)
    )
    g = sns.relplot(
        data=df,
        x="step",
        y="score",
        col="Eval",
        col_order=list(tasks.values()),
        kind="line",
        errorbar=None,
        height=3.4,
        aspect=1,
        **style,
    )

    # The base model is unattacked: draw it flat, as in plot_trajectory.
    base = {label: score for _, label, score in results("z7b-base")}
    for label, ax in g.axes_dict.items():
        ax.axhline(base[label], color="black")
    legend = dict(zip([t.get_text() for t in g.legend.texts], g.legend.legend_handles))
    if title := g.legend.get_title().get_text():  # a single group is drawn as title
        legend = {title: Line2D([], [], linestyle=""), **legend}
    method = {"Method": Line2D([], [], linestyle=""), "Base": Line2D([], [], color="k")}
    if "hue" not in style:  # single-colour plots: every run is TAR
        method["TAR"] = Line2D([], [], color=g.axes.flat[0].lines[0].get_color())
    g.legend.remove()
    g.add_legend(legend_data={**method, **legend})

    g.set_axis_labels("Relearning step", "Accuracy (%)")
    os.makedirs(osp.dirname(output), exist_ok=True)
    g.figure.savefig(output, bbox_inches="tight")


if __name__ == "__main__":
    main()
