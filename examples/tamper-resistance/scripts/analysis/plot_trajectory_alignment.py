import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

rootdir = "../booster/artifacts/mila/evals/figure3"
figure_dir = "artifacts/figures"
os.makedirs(figure_dir, exist_ok=True)

configs = [
    ("mixed-fresh-lora", "100 harmful + 900 SST-2"),
    ("harmful-only-fresh-lora", "1,000 harmful"),
    ("harmful-100-fresh-lora", "100 harmful × 100 epochs"),
]

rows = []
for experiment, data in configs:
    for path in glob.glob(osp.join(rootdir, experiment, "*", "step-*", "summary.json")):
        with open(path) as f:
            result = json.load(f)
        rows.append(
            {
                "step": result["step"],
                "score": 100 * result["harmful_score"],
                "method": result["method"].upper(),
                "data": data,
                "eval": "Harmful responses ↓",
            }
        )
        mmlu_path = osp.join(osp.dirname(path), "mmlu.json")
        if osp.isfile(mmlu_path):
            with open(mmlu_path) as f:
                mmlu = json.load(f)
            rows.append(
                {
                    "step": mmlu["step"],
                    "score": 100 * mmlu["mmlu_acc"],
                    "method": mmlu["method"].upper(),
                    "data": data,
                    "eval": "MMLU ↑",
                }
            )

df = pd.DataFrame(rows)
g = sns.relplot(
    data=df,
    x="step",
    y="score",
    col="eval",
    hue="method",
    style="data",
    markers=True,
    dashes=False,
    kind="line",
    errorbar=None,
)
g.set_axis_labels("Attack step", "Score (%)")
g.figure.savefig(osp.join(figure_dir, "alignment-trajectory.pdf"), bbox_inches="tight")
g.figure.savefig(
    osp.join(figure_dir, "alignment-trajectory.png"), dpi=200, bbox_inches="tight"
)
