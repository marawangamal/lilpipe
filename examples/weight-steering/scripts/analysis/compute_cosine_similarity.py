#!/usr/bin/env python3
"""Compare seeded Honesty, non-sycophancy, and non-cheating LoRA directions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from steering.steering_cones import (  # noqa: E402
    ConeAnalysisError,
    EffectiveVector,
    gram_matrix,
    load_effective_vector,
    low_rank_inner,
)


GROUPS = ("Honesty", "NS", "NC")
_LAYER_NUMBER = re.compile(r"(?:^|\.)(?:layers?|h)\.(\d+)(?:\.|$)")


def load_vector_specs(path: Path) -> list[dict[str, Any]]:
    """Build the fixed comparison vectors from the shared model registry."""

    try:
        raw = yaml.safe_load(path.read_text())
        models = raw["models"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as error:
        raise ConeAnalysisError(f"Could not read model registry {path}: {error}") from error
    pairs = (
        ("Honesty", "Honest", "Dishonest"),
        ("NS", "Non-Sycophantic", "Sycophantic"),
        ("NC", "Non-Cheat", "Cheat"),
    )
    specs = []
    for group, positive, negative in pairs:
        for seed in range(42, 47):
            positive_id = f"SmolLM3-3B-HMO-FT-{positive}-Seed-{seed}"
            negative_id = f"SmolLM3-3B-HMO-FT-{negative}-Seed-{seed}"
            try:
                positive_path = models[positive_id]["artifact"]
                negative_path = models[negative_id]["artifact"]
            except (KeyError, TypeError) as error:
                raise ConeAnalysisError(
                    f"Registry is missing an artifact for {positive_id} or {negative_id}"
                ) from error
            specs.append({
                "id": f"{group.lower()}-{seed}",
                "behavior": group,
                "seed": seed,
                "positive_adapter": positive_path,
                "negative_adapter": negative_path,
            })
    return specs


def _distribution(values: list[float]) -> dict[str, Any]:
    """Return raw values and compact descriptive statistics."""

    if not values:
        raise ConeAnalysisError("Cannot summarize an empty cosine distribution")
    return {
        "values": values,
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def cosine_distributions(
    vectors: list[EffectiveVector], cosine: np.ndarray
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Summarize all distinct within-group and Cartesian cross-group pairs."""

    indices = {group: [i for i, vector in enumerate(vectors) if vector.behavior == group] for group in GROUPS}
    within = {}
    for group, group_indices in indices.items():
        values = [float(cosine[left, right]) for offset, left in enumerate(group_indices) for right in group_indices[offset + 1:]]
        within[group] = _distribution(values)
    cross = {}
    for left_group, right_group in (("Honesty", "NS"), ("Honesty", "NC"), ("NS", "NC")):
        values = [float(cosine[left, right]) for left in indices[left_group] for right in indices[right_group]]
        cross[f"{left_group}-{right_group}"] = _distribution(values)
    return within, cross


def _layer_label(module: str) -> str:
    match = _LAYER_NUMBER.search(module)
    return match.group(1) if match else module


def layer_magnitudes(vector: EffectiveVector) -> dict[str, float]:
    """Compute effective-delta Frobenius norms grouped by transformer layer."""

    by_layer: dict[str, list] = {}
    for term in vector.terms:
        by_layer.setdefault(_layer_label(term.module), []).append(term)
    result = {}
    for layer, terms in by_layer.items():
        restricted = EffectiveVector(vector.id, vector.behavior, vector.seed, tuple(terms))
        result[layer] = math.sqrt(max(0.0, low_rank_inner(restricted, restricted)))
    return result


def analyze(vectors: list[EffectiveVector]) -> dict[str, Any]:
    """Compute norms, the full cosine matrix, distributions, and layer norms."""

    gram = gram_matrix(vectors)
    squared_norms = np.diag(gram)
    if np.any(squared_norms <= 0):
        bad = [vectors[i].id for i in np.flatnonzero(squared_norms <= 0)]
        raise ConeAnalysisError(f"Zero-norm effective vectors: {bad}")
    norms = np.sqrt(squared_norms)
    cosine = np.clip(gram / np.outer(norms, norms), -1.0, 1.0)
    within, cross = cosine_distributions(vectors, cosine)
    return {
        "vectors": [
            {
                "id": vector.id,
                "group": vector.behavior,
                "seed": vector.seed,
                "norm": float(norms[index]),
                "layer_magnitudes": layer_magnitudes(vector),
            }
            for index, vector in enumerate(vectors)
        ],
        "cosine_similarity": {
            "ids": [vector.id for vector in vectors],
            "matrix": cosine.tolist(),
        },
        "within_group_cosines": within,
        "cross_group_cosines": cross,
    }


def _save(figure: Any, output_dir: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}.pdf")


def plot_report(report: dict[str, Any], output_dir: Path) -> None:
    """Write the cosine heatmap and layer-wise panels."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = report["cosine_similarity"]["ids"]
    matrix = np.asarray(report["cosine_similarity"]["matrix"])
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    lower, upper = float(off_diagonal.min()), float(off_diagonal.max())
    padding = max(0.02, (upper - lower) * 0.05)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, vmin=max(-1, lower - padding), vmax=min(1, upper + padding), cmap="coolwarm")
    axis.set_xticks(range(15), ids, rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(15), ids, fontsize=8)
    for boundary in (4.5, 9.5):
        axis.axhline(boundary, color="black", linewidth=1.5)
        axis.axvline(boundary, color="black", linewidth=1.5)
    axis.set_title("Seeded effective-vector cosine similarity (zoomed)")
    figure.colorbar(image, ax=axis, label="cosine")
    _save(figure, output_dir, "cosine-similarity")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, group in zip(axes, GROUPS, strict=True):
        for vector in (item for item in report["vectors"] if item["group"] == group):
            magnitudes = vector["layer_magnitudes"]
            ordered = sorted(magnitudes, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
            axis.plot(range(len(ordered)), [magnitudes[layer] for layer in ordered], label=str(vector["seed"]), alpha=0.85)
        axis.set_title(group)
        axis.set_xlabel("layer (model order)")
        axis.legend(title="seed", fontsize=8)
    axes[0].set_ylabel("effective-delta Frobenius norm")
    figure.suptitle("Layer-wise contrastive LoRA magnitude")
    _save(figure, output_dir, "layerwise-norms")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    root = Path.cwd()
    vectors = [load_effective_vector(entry, root) for entry in load_vector_specs(Path(args.registry))]
    report = analyze(vectors)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, output_dir)


if __name__ == "__main__":
    main()
