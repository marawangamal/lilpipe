import argparse
import glob
import json
import os
import os.path as osp
import re

import pandas as pd
import seaborn as sns

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

P = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar"
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
            style="Attack lr",
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
            (f"{P}-k16--relearn-fft-lr1e-6-ms128-bs-4", "TAR-ME 16 steps", 1, "1e-6"),
            (f"{P}-k16--relearn-fft-lr5e-6-ms128-bs-4", "TAR-ME 16 steps", 1, "5e-6"),
        ],
        dict(
            size="Attack lr",
            sizes={"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
            size_order=["1e-6", "5e-6", "1e-5"],
            style="Batch size",
            markers=True,
            dashes=False,
        ),
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--plot", choices=plots, default="tar")
parser.add_argument("--output")
args = parser.parse_args()
configs, style = plots[args.plot]
output = args.output or (
    f"artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--{args.plot}.pdf"
)

rows = []
for experiment, defence, seed, lr in configs:
    for path in glob.glob(
        osp.join(rootdir, experiment, "checkpoint-*", "**", "results*.json"),
        recursive=True,
    ):
        step = int(path.split("checkpoint-")[1].split("/")[0])
        with open(path) as f:
            results = json.load(f)["results"]
        for task, label in tasks.items():
            if task in results:
                rows.append(
                    {
                        "step": step,
                        "score": 100 * results[task]["acc,none"],
                        "Defence": defence,
                        "Seed": seed,
                        "Attack lr": lr,
                        "Attack steps": (
                            ms[1]
                            if (ms := re.search(r"-ms(\d+)", experiment))
                            else "16"
                        ),
                        "Batch size": (
                            bs[1]
                            if (bs := re.search(r"-bs-(\d+)", experiment))
                            else "1"
                        ),
                        "Eval": label,
                    }
                )

df = pd.DataFrame(rows)
g = sns.relplot(
    data=df,
    x="step",
    y="score",
    col="Eval",
    col_order=list(tasks.values()),
    kind="line",
    errorbar=None,
    **style,
)
# The base model is unattacked: draw it flat across all steps, as in plot_trajectory.
base = {}
for path in glob.glob(
    osp.join(rootdir, "z7b-base", "checkpoint-0", "**", "results*.json"), recursive=True
):
    with open(path) as f:
        results = json.load(f)["results"]
    base.update(
        {
            label: 100 * results[t]["acc,none"]
            for t, label in tasks.items()
            if t in results
        }
    )
steps = sorted(df["step"].unique())
for label, ax in g.axes_dict.items():
    (base_line,) = ax.plot(
        steps, [base[label]] * len(steps), color="black", marker="o", markersize=4
    )
legend = dict(zip([t.get_text() for t in g.legend.texts], g.legend.legend_handles))
g.legend.remove()
g.add_legend(legend_data={"Base": base_line, **legend})
g.set_axis_labels("Relearning step", "Accuracy (%)")
os.makedirs(osp.dirname(output), exist_ok=True)
g.figure.savefig(output, bbox_inches="tight")
