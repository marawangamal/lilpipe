#!/usr/bin/env python3
"""Plot WMDP relearning trajectories; uncomment one selection block below."""

import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

tar = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
seam = "z7b-wmdp-bio--5pp--unlearn-fft-gd-seam"
configs_main = [
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft",
        "method": "RMU",
        "defence": "None",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
        "method": "RMU",
        "defence": "SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-rmu-wd--relearn-fft",
        "method": "RMU",
        "defence": "Weight distance",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo--relearn-fft",
        "method": "NPO",
        "defence": "None",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
        "method": "NPO",
        "defence": "SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-wd--relearn-fft-s2",
        "method": "NPO",
        "defence": "Weight distance",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft",
        "method": "MaxEnt",
        "defence": "None",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft",
        "method": "MaxEnt",
        "defence": "SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me-wd--relearn-fft",
        "method": "MaxEnt",
        "defence": "Weight distance",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft",
        "method": "MaxEnt",
        "defence": "TAR",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-gd-seam--relearn-fft",
        "method": "GD",
        "defence": "SEAM",
    },
]

configs_npo_sam_15pp = [
    {
        "model": "z7b-wmdp-bio--15pp--unlearn-fft-npo--relearn-fft",
        "method": "NPO",
    },
    {
        "model": "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
        "method": "NPO + SAM",
    },
]


configs_npo_sam_15pp_gt_bs2 = [
    {
        "model": "z7b-wmdp-bio--15pp--unlearn-fft-npo-gt--relearn-fft-gt-bs2",
        "method": "NPO",
    },
    {
        "model": "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2-gt--relearn-fft-gt-bs2",
        "method": "NPO + SAM",
    },
]


configs_npo_sam_gip_15pp_gt_bs2 = configs_npo_sam_15pp_gt_bs2 + [
    {
        "model": "z7b-wmdp-bio--15pp--unlearn-fft-npo-gip-lambda1-gt--relearn-fft-gt-bs2",
        "method": "NPO + GIP (λ=1, BF16)",
    },
]

configs_npo_sam_15pp_seeds = [
    {
        "model": f"{method}/eval",
        "evaluation_root": "artifacts/tamia/runs/npo-refactor-fix-d26b5d81",
        "method": label,
        "seed": 42,
    }
    for method, label in [("npo", "NPO"), ("npo-sam-rho1e-2", "NPO + SAM")]
] + [
    {
        "model": f"z7b-wmdp-bio--15pp--unlearn-fft-{method}-gt--seed{seed}--relearn-fft-gt-bs2",
        "method": label,
        "seed": seed,
    }
    for method, label in [("npo", "NPO"), ("npo-sam-rho1e-2", "NPO + SAM")]
    for seed in [43, 44]
]

configs_lrs = [
    {"model": f"{model}--relearn-fft-lr{lr}-ms128", "method": method, "lr": lr}
    for model, method in ((tar, "TAR"), (seam, "SEAM"))
    for lr in ("1e-6", "5e-6", "1e-5")
]
configs_tar_lrs = [
    {
        "model": f"{tar}--relearn-fft-ms128--hpo--grid-001--{index:03d}",
        "method": "TAR",
        "lr": lr,
        "baseline": tar,
    }
    for index, lr in enumerate(("1e-6", "5e-6", "1e-5"), 1)
]
configs_ms1000 = [
    {"model": f"{tar}--relearn-fft-lr{lr}-ms1000", "method": "TAR", "lr": lr}
    for lr in ("1e-6", "5e-6", "1e-5")
]
configs_perturbed_ms1000 = [
    {
        "model": f"{tar}--relearn-perturb-and-fft-sigma{sigma}-lr1e-5-ms1000",
        "method": "TAR",
        "sigma": sigma,
    }
    for sigma in ("1e-4", "1e-3", "1e-2", "1e-1")
]


configs_npo_sam_gip = [
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-gt--relearn-fft-gt",
        "method": "NPO",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2-gt--relearn-fft-gt",
        "method": "NPO + SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-gip-lambda1-gt--relearn-fft-gt",
        "method": "NPO + GIP",
    },
]


configs_npo_sam_gip_lrs = [
    {
        "model": config["model"].replace("--relearn-fft-gt", suffix),
        "method": config["method"],
        "lr": lr,
    }
    for config in configs_npo_sam_gip
    for lr, suffix in (
        ("1e-5", "--relearn-fft-gt"),
        ("1e-6", "--relearn-fft-gt-lr1e-6-ms16"),
        ("5e-6", "--relearn-fft-gt-lr5e-6-ms16"),
    )
]


def results(experiment, baseline=None, root=rootdir):
    if baseline is not None:
        for _, label, score in results(baseline, root=root):
            yield 0, label, score
    pattern = osp.join(root, experiment, "checkpoint-*", "**", "results*.json")
    paths = glob.glob(pattern, recursive=True, include_hidden=True)
    if not paths:
        paths = glob.glob(
            osp.join(root, experiment, "**", "results*.json"),
            recursive=True,
            include_hidden=True,
        )
    for path in sorted(
        paths,
        key=lambda path: (
            int(path.split("checkpoint-")[1].split(os.sep)[0])
            if "checkpoint-" in path
            else 0
        ),
    ):
        with open(path) as handle:
            result = json.load(handle)["results"]
        step = (
            int(path.split("checkpoint-")[1].split(os.sep)[0])
            if "checkpoint-" in path
            else 0
        )
        for task, label in tasks.items():
            if task in result:
                yield step, label, 100 * result[task]["acc,none"]


def trajectory_rows(configs, steps, require_complete=False):
    rows = []
    expected = {(step, label) for step in steps for label in tasks.values()}
    for config in configs:
        observations = [
            (step, evaluation, accuracy)
            for step, evaluation, accuracy in results(
                config["model"],
                config.get("baseline"),
                root=config.get("evaluation_root", rootdir),
            )
            if step in steps
        ]
        actual = {(step, evaluation) for step, evaluation, _ in observations}
        if require_complete and (
            actual != expected or len(observations) != len(expected)
        ):
            raise ValueError(
                f"Incomplete or duplicate trajectory: {config['model']}; "
                f"missing {sorted(expected - actual)}"
            )
        for step, evaluation, accuracy in observations:
            rows.append(
                {
                    "Relearning step": step,
                    "Accuracy (%)": accuracy,
                    "Method": config["method"],
                    "Defence": config.get("defence", ""),
                    "LR": config.get("lr", ""),
                    "σ": config.get("sigma", ""),
                    "Seed": config.get("seed", ""),
                    "Eval": evaluation,
                }
            )
    return rows


def legend_with_base(legend, base):
    """Place Base inside the Method section when that section exists."""
    if "Method" in legend:
        method = legend.pop("Method")
        return {"Method": method, "Base": base, **legend}
    return {"Base": base, **legend}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--tar-lrs", action="store_true")
    parser.add_argument("--npo-sam-seeds", action="store_true")
    args = parser.parse_args()

    # Uncomment one complete selection block.
    # configs = configs_main
    # steps = range(0, 17, 4)
    # style = {
    #     "hue": "Method",
    #     "style": "Defence",
    #     "markers": True,
    #     "dashes": False,
    #     "palette": {
    #         "RMU": "#4477AA",
    #         "NPO": "#EE6677",
    #         "MaxEnt": "#228833",
    #         "GD": "#AA3377",
    #     },
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf"

    # configs = configs_npo_sam_15pp
    # steps = range(0, 17, 4)
    # style = {"hue": "Method", "marker": "o", "dashes": False}
    # output = (
    #     "artifacts/figures/"
    #     "z7b-wmdp-mmlu-relearning-trajectory--15pp--npo-sam-reproduction.pdf"
    # )

    configs = configs_npo_sam_gip_15pp_gt_bs2
    steps = range(0, 17, 4)
    style = {"hue": "Method", "marker": "o", "dashes": False}
    output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--15pp--bs2.pdf"

    # configs = configs_npo_sam_gip
    # steps = range(0, 17, 4)
    # style = {"hue": "Method", "marker": "o", "dashes": False}
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--npo-sam-gip-gt-ms16.pdf"

    # configs = configs_npo_sam_gip_lrs
    # steps = range(0, 17, 4)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.9, "5e-6": 1.7, "1e-5": 2.5},
    #     "marker": "o",
    #     "dashes": False,
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--npo-sam-gip-gt-lrs-ms16.pdf"

    # configs = configs_lrs
    # steps = range(0, 129)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--ms128.pdf"

    # configs = configs_tar_lrs
    # steps = range(0, 129)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--tar-k16-lrs.pdf"

    # configs = configs_ms1000
    # steps = range(0, 1001)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--ms1000.pdf"

    # configs = configs_perturbed_ms1000
    # steps = range(0, 1001)
    # style = {
    #     "hue": "Method",
    #     "size": "σ",
    #     "size_order": ["1e-4", "1e-3", "1e-2", "1e-1"],
    #     "sizes": {"1e-4": 0.6, "1e-3": 1.2, "1e-2": 2.0, "1e-1": 3.0},
    # }
    # output = (
    #     "artifacts/figures/"
    #     "z7b-wmdp-mmlu-relearning-trajectory-perturbed--ms1000.pdf"
    # )

    if args.tar_lrs:
        configs = configs_tar_lrs
        steps = range(0, 129, 16)
        style = {
            "hue": "LR",
            "hue_order": ["1e-6", "5e-6", "1e-5"],
            "markers": True,
            "marker": "o",
            "dashes": False,
        }
        output = (
            "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--tar-k16-new-api.pdf"
        )

    if args.npo_sam_seeds:
        configs = configs_npo_sam_15pp_seeds
        steps = range(0, 17, 4)
        style = {
            "hue": "Method",
            "marker": "o",
            "dashes": False,
            "palette": {"NPO": "#4477AA", "NPO + SAM": "#EE6677"},
        }
        output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--15pp--bs2--seeds.pdf"

    rows = trajectory_rows(configs, steps, require_complete=args.npo_sam_seeds)
    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Accuracy (%)",
        col="Eval",
        col_order=list(tasks.values()),
        kind="line",
        errorbar="sd" if args.npo_sam_seeds else None,
        height=3.4,
        aspect=1,
        **style,
    )

    base = {label: score for _, label, score in results("z7b-base")}
    for label, axis in grid.axes_dict.items():
        if args.npo_sam_seeds:
            axis.set_xticks(list(steps))
            sns.lineplot(
                data=df[df["Eval"] == label],
                x="Relearning step",
                y="Accuracy (%)",
                hue="Method",
                units="Seed",
                estimator=None,
                palette=style["palette"],
                alpha=0.25,
                linewidth=0.8,
                legend=False,
                ax=axis,
            )
        if label in base:
            axis.axhline(base[label], color="black")

    legend = dict(
        zip(
            [text.get_text() for text in grid.legend.texts],
            grid.legend.legend_handles,
        )
    )
    if base:
        legend = legend_with_base(legend, Line2D([], [], color="black"))
    grid.legend.remove()
    grid.add_legend(legend_data=legend)
    grid.set_axis_labels("Relearning step", "Accuracy (%)")
    if args.npo_sam_seeds:
        grid.figure.suptitle("Mean ±1 SD across three defense seeds", y=1.03)
    output = args.output or output
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")
