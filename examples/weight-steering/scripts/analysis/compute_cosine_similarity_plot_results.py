#!/usr/bin/env python3
"""Plot seeded cosine-similarity results from a generated JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


GROUPS = ("Honesty", "NS", "NC")


def _save(figure: Any, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.pdf")


def plot_report(report: dict[str, Any], output_dir: Path) -> None:
    """Write the cosine heatmap and layer-wise panels."""

    ids = report["cosine_similarity"]["ids"]
    matrix = np.asarray(report["cosine_similarity"]["matrix"])
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    lower, upper = float(off_diagonal.min()), float(off_diagonal.max())
    padding = max(0.02, (upper - lower) * 0.05)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(
        matrix,
        vmin=max(-1, lower - padding),
        vmax=min(1, upper + padding),
        cmap="coolwarm",
    )
    axis.set_xticks(range(len(ids)), ids, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(len(ids)), ids, fontsize=8)
    for boundary in (4.5, 9.5):
        axis.axhline(boundary, color="black", linewidth=1.5)
        axis.axvline(boundary, color="black", linewidth=1.5)
    axis.set_title("Seeded effective-vector cosine similarity (zoomed)")
    figure.colorbar(image, ax=axis, label="cosine")
    _save(figure, output_dir, "cosine-similarity")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, group in zip(axes, GROUPS, strict=True):
        for vector in (
            item for item in report["vectors"] if item["group"] == group
        ):
            magnitudes = vector["layer_magnitudes"]
            ordered = sorted(
                magnitudes,
                key=lambda value: (
                    not value.isdigit(),
                    int(value) if value.isdigit() else value,
                ),
            )
            axis.plot(
                range(len(ordered)),
                [magnitudes[layer] for layer in ordered],
                label=str(vector["seed"]),
                alpha=0.85,
            )
        axis.set_title(group)
        axis.set_xlabel("layer (model order)")
        axis.legend(title="seed", fontsize=8)
    axes[0].set_ylabel("effective-delta Frobenius norm")
    figure.suptitle("Layer-wise contrastive LoRA magnitude")
    _save(figure, output_dir, "layerwise-norms")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output_dir", nargs="?")
    args = parser.parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(results_path.read_text())
    plot_report(report, output_dir)


if __name__ == "__main__":
    main()
