"""Plot one-epoch relearning; cosine measurements are available before attack only."""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="artifacts/figures/figure3-npo-gip-samples.csv",
    )
    parser.add_argument("--output", default="artifacts/figures/figure3-npo-gip-samples")
    args = parser.parse_args()

    data = pd.read_csv(args.results)

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    names = data.Method.drop_duplicates().tolist()
    palette = dict(zip(names, sns.color_palette(n_colors=len(names))))
    markers = dict(zip(names, ["o", "s", "^", "D"]))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for axis, (metric, ylabel) in zip(
        axes,
        [
            ("Utility", "Full MMLU accuracy (%)"),
            ("WMDP", "WMDP-Bio accuracy (%)"),
            ("Cosine similarity", "Mean off-diagonal gradient cosine"),
        ],
    ):
        subset = data[data.Metric == metric]
        if metric == "Cosine similarity":
            sns.scatterplot(
                data=subset,
                x="Samples",
                y="Value",
                hue="Method",
                hue_order=names,
                palette=palette,
                style="Method",
                style_order=names,
                markers=markers,
                s=65,
                legend=False,
                ax=axis,
            )
            axis.text(
                0.98,
                0.96,
                "Before attack only · BF16",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            axis.set_ylim(-0.05, 1.05)
        else:
            sns.lineplot(
                data=subset,
                x="Samples",
                y="Value",
                hue="Method",
                hue_order=names,
                palette=palette,
                style="Method",
                style_order=names,
                markers=markers,
                dashes=False,
                errorbar="se",
                legend=(metric == "Utility"),
                ax=axis,
            )
        axis.set(title=metric, xlabel="Relearning samples (1 epoch)", ylabel=ylabel)
        axis.set_xticks([0, 20, 40, 60])
        axis.set_xlim(-3, 63)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    for extension in ["pdf", "png"]:
        figure.savefig(args.output + "." + extension, bbox_inches="tight", dpi=180)
    print(args.output + ".pdf")
