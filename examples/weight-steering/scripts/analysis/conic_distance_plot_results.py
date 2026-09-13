#!/usr/bin/env python3
"""Plot seeded cone-distance results from a generated JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def plot_report(report: Any, output_dir: Path) -> None:
    """Write the annotated target-to-behavior-cone heatmap."""

    matrix = np.asarray(report, dtype=float)
    if matrix.shape != (15, 3):
        raise ValueError("expected a 15x3 cone-distance matrix")

    row_ids = [
        f"{behavior}-{seed}"
        for behavior in ("honesty", "ns", "nc")
        for seed in range(42, 47)
    ]
    figure, axis = plt.subplots(figsize=(6, 9))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(3), ("honesty", "ns", "nc"))
    axis.set_yticks(range(15), row_ids, fontsize=8)
    axis.set_xlabel("behavior cone")
    axis.set_ylabel("target steering vector")
    axis.set_title("Seed-to-cone distance")
    for boundary in (4.5, 9.5):
        axis.axhline(boundary, color="white", linewidth=1.5)
    for row in range(15):
        for column in range(3):
            value = matrix[row, column]
            color = "white" if value < 0.55 else "black"
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center", color=color
            )
    figure.colorbar(image, ax=axis, label="relative cone distance")
    figure.tight_layout()
    figure.savefig(output_dir / "conic-distance.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output_dir", nargs="?")
    args = parser.parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_report(json.loads(results_path.read_text()), output_dir)


if __name__ == "__main__":
    main()
