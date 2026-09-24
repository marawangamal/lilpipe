#!/usr/bin/env python3
"""Plot dense LoRA update gradient agreement across unlearning and relearning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_results(
    source: Path, output: Path, methods: tuple[str, ...] = ("cb", "npo")
) -> None:
    import matplotlib.pyplot as plt

    rows = json.loads(source.read_text())["rows"]
    figure, axis = plt.subplots(figsize=(9, 5))
    colors = {"cb": "tab:blue", "npo": "tab:orange"}
    for method in methods:
        color = colors[method]
        for stage, linestyle in (("unlearn", "-"), ("relearn", "--")):
            trajectory = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method and row["stage"] == stage
                ),
                key=lambda row: row["cumulative_step"],
            )
            if trajectory:
                axis.plot(
                    [row["cumulative_step"] for row in trajectory],
                    [row["mean_cosine"] for row in trajectory],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    label=f"{method.upper()} {stage}",
                )
    axis.axvline(32, color="gray", linestyle=":", label="Relearning begins")
    axis.set(
        xlabel="Cumulative optimizer step",
        ylabel="Mean pairwise dense-update gradient cosine",
        title="WMDP-Bio forget-document gradient agreement",
        xlim=(0, 65),
        ylim=(-1, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/analysis/per_sample_param_grad_cosim.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/per_sample_param_grad_cosim.png"),
    )
    parser.add_argument("--method", choices=("all", "cb", "npo"), default="all")
    args = parser.parse_args()
    methods = ("cb", "npo") if args.method == "all" else (args.method,)
    plot_results(args.input, args.output, methods)
    print(args.output)


if __name__ == "__main__":
    main()
