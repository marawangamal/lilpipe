#!/usr/bin/env python3
"""Compute distances from seeded steering vectors to behavior cones."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from cosine_similarity_gen_results import SEEDS, dot

BEHAVIORS = (
    ("Honest", "Dishonest"),
    ("Non-Sycophantic", "Sycophantic"),
    ("Non-Cheat", "Cheat"),
)
GROUP_SIZE = len(SEEDS)


def build_steer_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Return steering adapter pairs in behavior-major, seed-minor order."""

    return [
        (
            root / f"SmolLM3-3B-HMO-FT-{positive}-Seed-{seed}",
            root / f"SmolLM3-3B-HMO-FT-{negative}-Seed-{seed}",
        )
        for positive, negative in BEHAVIORS
        for seed in SEEDS
    ]


def normalized_gram(steer_pairs: Sequence[tuple[Path, Path]]) -> np.ndarray:
    """Build the normalized Gram matrix without materializing dense updates."""

    size = len(steer_pairs)
    raw = np.empty((size, size), dtype=float)
    for i in range(size):
        for j in range(i, size):
            raw[i, j] = raw[j, i] = dot(steer_pairs[i], steer_pairs[j])

    norms = np.sqrt(np.diag(raw))
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("all steering vectors must have a finite, positive norm")
    gram = raw / np.outer(norms, norms)
    gram = (gram + gram.T) / 2
    np.fill_diagonal(gram, 1.0)
    return gram


def cone_distance(
    generator_gram: np.ndarray,
    target_dots: np.ndarray,
    target_norm_squared: float,
) -> float:
    """Return the distance to a cone using only vector inner products."""

    result = minimize(
        fun=lambda coefficients: (
            coefficients @ generator_gram @ coefficients
            - 2.0 * coefficients @ target_dots
        ),
        jac=lambda coefficients: (
            2.0 * generator_gram @ coefficients - 2.0 * target_dots
        ),
        x0=np.zeros(len(target_dots)),
        bounds=[(0.0, None)] * len(target_dots),
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(result.message)

    coefficients = result.x
    distance_squared = float(
        target_norm_squared
        - 2.0 * coefficients @ target_dots
        + coefficients @ generator_gram @ coefficients
    )
    return float(np.sqrt(max(distance_squared, 0.0)))


def cone_distance_from_gram(
    gram: np.ndarray, target_index: int, generator_indices: Sequence[int]
) -> float:
    """Extract a cone projection problem from a complete Gram matrix."""

    generators = np.asarray(tuple(generator_indices), dtype=int)
    return cone_distance(
        gram[np.ix_(generators, generators)],
        gram[generators, target_index],
        float(gram[target_index, target_index]),
    )


def cone_generator_indices(
    target_index: int, behavior_index: int, group_size: int = GROUP_SIZE
) -> tuple[int, ...]:
    """Select all seeds that generate a behavior cone."""

    start = behavior_index * group_size
    return tuple(range(start, start + group_size))


def compute_conic_distances(
    gram: np.ndarray, group_size: int = GROUP_SIZE
) -> np.ndarray:
    """Compute target-to-behavior-cone distances for a normalized Gram matrix."""

    group_count = len(BEHAVIORS)
    expected_size = group_count * group_size
    if gram.shape != (expected_size, expected_size):
        raise ValueError(f"expected a {expected_size}x{expected_size} Gram matrix")
    return np.asarray(
        [
            [
                cone_distance_from_gram(
                    gram,
                    target_index,
                    cone_generator_indices(target_index, behavior_index, group_size),
                )
                for behavior_index in range(group_count)
            ]
            for target_index in range(expected_size)
        ]
    )


def main() -> None:
    pairs = build_steer_pairs(Path("artifacts/models"))
    distances = compute_conic_distances(normalized_gram(pairs))
    output = Path("artifacts/analysis/conic-distance-across-seeds/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(distances.tolist(), indent=2) + "\n")


if __name__ == "__main__":
    main()
