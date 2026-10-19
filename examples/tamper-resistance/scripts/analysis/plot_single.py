import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

P = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar"
# name: (runs as (eval dir, defence, seed, attack lr), hue)
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
        "defence",
    ),
    "tar-long": (
        [
            (f"{P}-k16--relearn-fft-lr5e-6-ms64", "TAR-ME 16 steps", 1, "5e-6"),
            (f"{P}-k16--relearn-fft-lr1e-5-ms64", "TAR-ME 16 steps", 1, "1e-5"),
        ],
        "attack lr",
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--plot", choices=plots, default="tar")
parser.add_argument("--output")
args = parser.parse_args()
configs, hue = plots[args.plot]
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
                        "defence": defence,
                        "seed": seed,
                        "attack lr": lr,
                        "eval": label,
                    }
                )

df = pd.DataFrame(rows)
g = sns.relplot(
    data=df,
    x="step",
    y="score",
    col="eval",
    col_order=list(tasks.values()),
    hue=hue,
    size="seed",
    sizes={1: 0.6, 2: 1.5},
    style="attack lr" if hue == "defence" else "defence",
    markers=True,
    dashes=False,
    kind="line",
    errorbar=None,
)
g.set_axis_labels("Relearning step", "Accuracy (%)")
os.makedirs(osp.dirname(output), exist_ok=True)
g.figure.savefig(output, bbox_inches="tight")
