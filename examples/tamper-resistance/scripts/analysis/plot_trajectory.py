import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

rootdir = "artifacts/tamia/evals"
figure_dir = "artifacts/figures"
os.makedirs(figure_dir, exist_ok=True)

configs = [
    {
        "path": osp.join(rootdir, "z7b-wmdp-bio--unlearn-fft-gd--relearn-fft-long"),
        "unlearning_method": "GD",
        "tamper_resistance_method": "None",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2--relearn-fft-long",
        ),
        "unlearning_method": "GD",
        "tamper_resistance_method": "SAM",
    },
    {
        "path": osp.join(rootdir, "z7b-wmdp-bio--unlearn-fft-gd-ao--relearn-fft-long"),
        "unlearning_method": "GD",
        "tamper_resistance_method": "AO",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-gd--resist-fft-weight-distance--relearn-fft-long",
        ),
        "unlearning_method": "GD",
        "tamper_resistance_method": "Weight distance",
    },
    {
        "path": osp.join(rootdir, "z7b-wmdp-bio--unlearn-fft-npo--relearn-fft-long"),
        "unlearning_method": "NPO",
        "tamper_resistance_method": "None",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-npo-sam-rho1e-3--relearn-fft-long",
        ),
        "unlearning_method": "NPO",
        "tamper_resistance_method": "SAM",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-npo-weight-distance--relearn-fft-long",
        ),
        "unlearning_method": "NPO",
        "tamper_resistance_method": "Weight distance",
    },
    {
        "path": osp.join(rootdir, "z7b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long"),
        "unlearning_method": "RMU",
        "tamper_resistance_method": "None",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-3--relearn-fft-long",
        ),
        "unlearning_method": "RMU",
        "tamper_resistance_method": "SAM",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft-long",
        ),
        "unlearning_method": "RMU",
        "tamper_resistance_method": "Weight distance",
    },
]

tasks = {
    "wmdp_bio_robust": "wmdp_bio_robust ↓",
    "mmlu_no_bio": "mmlu_no_bio ↑",
}
steps = range(0, 126, 25)

rows = []
for config in configs:
    checkpoints = glob.glob(osp.join(config["path"], "checkpoint-*"))
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
                if task in results:
                    rows.append(
                        {
                            "Unlearning Method": config["unlearning_method"],
                            "Tamper Resistance Method": config[
                                "tamper_resistance_method"
                            ],
                            "step": step,
                            "eval": label,
                            "acc": 100 * results[task]["acc,none"],
                        }
                    )

df = pd.DataFrame(rows)
plot = sns.relplot(
    data=df,
    x="step",
    y="acc",
    col="eval",
    hue="Unlearning Method",
    style="Tamper Resistance Method",
    markers=True,
    dashes=False,
    kind="line",
    errorbar=None,
)
plot.set_axis_labels("Relearning step", "Accuracy (%)")
output = osp.join(figure_dir, "relearning-attack-capability-filtered")
plot.figure.savefig(f"{output}.pdf", bbox_inches="tight")
plot.figure.savefig(f"{output}.png", bbox_inches="tight", dpi=200)
