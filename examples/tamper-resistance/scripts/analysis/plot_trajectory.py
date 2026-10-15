import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

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
        "path": osp.join(
            rootdir, "z7b-wmdp-bio--unlearn-fft-gd-deff--relearn-fft-long"
        ),
        "unlearning_method": "GD",
        "tamper_resistance_method": "DEFF",
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
        "mmlu_path": osp.join(
            rootdir, "z7b-wmdp-bio--unlearn-fft-rmu", "checkpoint-150"
        ),
        "unlearning_method": "RMU",
        "tamper_resistance_method": "None",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-3--relearn-fft-long",
        ),
        "mmlu_path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-3",
            "checkpoint-150",
        ),
        "unlearning_method": "RMU",
        "tamper_resistance_method": "SAM",
    },
    {
        "path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft-long",
        ),
        "mmlu_path": osp.join(
            rootdir,
            "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance",
            "checkpoint-150",
        ),
        "unlearning_method": "RMU",
        "tamper_resistance_method": "Weight distance",
    },
]
base_model_path = osp.join(rootdir, "z7b-base", "checkpoint-0")
random_chance = 25
steps_to_plot = range(0, 126, 25)
max_initial_mmlu_drop = 2
failed_criterion_alpha = 0.25

name_to_name = {
    "wmdp_bio_robust": "wmdp_bio_robust ↓",
    "mmlu_no_bio": "mmlu_no_bio ↑",
}


def get_tasks(checkpoint):
    tasks = []
    for path in glob.glob(
        os.path.join(checkpoint, "**", "results*.json"), recursive=True
    ):
        with open(path) as f:
            results = json.load(f)["results"]

        for name in ("wmdp_bio_robust", "mmlu_no_bio"):
            if name in results:
                tasks.append(
                    {"name": name_to_name[name], "acc": 100 * results[name]["acc,none"]}
                )
    return tasks


rows = []
for config in configs:
    checkpoints = glob.glob(os.path.join(config["path"], "checkpoint-*"))
    checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))

    for checkpoint in checkpoints:
        step = int(os.path.basename(checkpoint).split("-")[-1])
        if step not in steps_to_plot:
            continue
        for task in get_tasks(checkpoint):
            rows.append(
                {
                    "unlearning_method": config["unlearning_method"],
                    "tamper_resistance_method": config["tamper_resistance_method"],
                    "step": step,
                    "eval": task["name"],
                    "acc": task["acc"],
                }
            )

    if "mmlu_path" in config:
        for task in get_tasks(config["mmlu_path"]):
            if task["name"] == "mmlu_no_bio ↑":
                rows.append(
                    {
                        "unlearning_method": config["unlearning_method"],
                        "tamper_resistance_method": config["tamper_resistance_method"],
                        "step": 0,
                        "eval": task["name"],
                        "acc": task["acc"],
                    }
                )

df = pd.DataFrame(rows)
base_model = {task["name"]: task["acc"] for task in get_tasks(base_model_path)}
all_unlearning_methods = list(dict.fromkeys(df["unlearning_method"]))
palette = dict(
    zip(
        all_unlearning_methods,
        sns.color_palette("colorblind", n_colors=len(all_unlearning_methods)),
    )
)
markers = {"None": "o", "DEFF": "D", "SAM": "s", "Weight distance": "^"}
initial_mmlu = df[(df["eval"] == name_to_name["mmlu_no_bio"]) & (df["step"] == 0)]
failed_criterion = {
    (row.unlearning_method, row.tamper_resistance_method)
    for row in initial_mmlu.itertuples()
    if base_model[name_to_name["mmlu_no_bio"]] - row.acc > max_initial_mmlu_drop
}


def plot_trajectories(plot_df, output_name, formats):
    unlearning_methods = list(dict.fromkeys(plot_df["unlearning_method"]))
    tamper_resistance_methods = list(dict.fromkeys(plot_df["tamper_resistance_method"]))
    g = sns.relplot(
        data=plot_df,
        x="step",
        y="acc",
        col="eval",
        hue="unlearning_method",
        hue_order=unlearning_methods,
        palette=palette,
        style="tamper_resistance_method",
        style_order=tamper_resistance_methods,
        markers=markers,
        dashes=False,
        kind="line",
        height=3,
        markersize=4,
        errorbar=None,
    )

    for ax in g.axes.flat:
        for line in ax.lines:
            for unlearning_method, tamper_resistance_method in failed_criterion:
                colors_match = all(
                    abs(actual - expected) < 1e-8
                    for actual, expected in zip(
                        to_rgba(line.get_color()), to_rgba(palette[unlearning_method])
                    )
                )
                if (
                    colors_match
                    and line.get_marker() == markers[tamper_resistance_method]
                ):
                    line.set_alpha(failed_criterion_alpha)

    for eval_name, ax in zip(g.col_names, g.axes.flat):
        acc = base_model[eval_name]
        ax.axhline(acc, color="black", linewidth=1)
        ax.axhline(random_chance, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 60)

    base_model_handle = Line2D([], [], color="black", label="base model")
    random_chance_handle = Line2D(
        [], [], color="gray", linestyle="--", label="random chance"
    )
    handles = [*g.legend.legend_handles, base_model_handle, random_chance_handle]
    labels = [
        *[text.get_text() for text in g.legend.texts],
        "base model",
        "random chance",
    ]
    g.legend.remove()
    g.figure.legend(handles, labels, title=None, loc="center right")

    output_path = osp.join(figure_dir, output_name)
    for extension, options in formats:
        g.figure.savefig(f"{output_path}.{extension}", bbox_inches="tight", **options)
    plt.close(g.figure)


plot_trajectories(
    df,
    "relearning-attack-capability-filtered",
    (("pdf", {}), ("png", {"dpi": 200})),
)
for method in all_unlearning_methods:
    plot_trajectories(
        df[df["unlearning_method"] == method],
        f"relearning-attack-{method.lower()}-family",
        (("pdf", {}),),
    )
