import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

rootdir = "artifacts/tamia/evals/thresholds"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

configs = [
    (
        "5pp/z7b-wmdp-bio--5pp--unlearn-fft-me-tar--relearn-fft",
        "TAR-ME 8 steps",
        1,
    ),
    (
        "5pp/z7b-wmdp-bio--5pp--unlearn-fft-me-tar--relearn-fft-s2",
        "TAR-ME 8 steps",
        2,
    ),
    (
        "5pp/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft",
        "TAR-ME 16 steps",
        1,
    ),
    (
        "5pp/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-s2",
        "TAR-ME 16 steps",
        2,
    ),
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", default="artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--tar.pdf"
)
args = parser.parse_args()

rows = []
for experiment, defence, seed in configs:
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
    hue="defence",
    size="seed",
    sizes={1: 0.6, 2: 1.5},
    markers=True,
    marker="o",
    kind="line",
    errorbar=None,
)
g.set_axis_labels("Relearning step", "Accuracy (%)")
os.makedirs(osp.dirname(args.output), exist_ok=True)
g.figure.savefig(args.output, bbox_inches="tight")
