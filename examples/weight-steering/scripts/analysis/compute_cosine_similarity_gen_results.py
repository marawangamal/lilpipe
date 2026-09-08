#!/usr/bin/env python3
"""Generate cosine similarities between seeded contrastive LoRA directions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from steering.steering_cones import (  # noqa: E402
    EffectiveVector,
    load_effective_vector,
    low_rank_inner,
)


SEEDS = range(42, 47)
PAIR_TEMPLATES = (
    ("Honesty", "honest-{seed}", "dishonest-{seed}"),
    ("NS", "non-sycophantic-{seed}", "sycophantic-{seed}"),
    ("NC", "non-cheat-{seed}", "cheat-{seed}"),
)
ADAPTER_ROOT = Path("artifacts/models/cosine-similarity-across-seeds")
STEER_PAIRS = [
    {
        "id": f"{group.lower()}-{seed}",
        "behavior": group,
        "seed": seed,
        "positive_adapter": str(ADAPTER_ROOT / positive.format(seed=seed)),
        "negative_adapter": str(ADAPTER_ROOT / negative.format(seed=seed)),
    }
    for group, positive, negative in PAIR_TEMPLATES
    for seed in SEEDS
]

_LAYER_NUMBER = re.compile(r"(?:^|\.)(?:layers?|h)\.(\d+)(?:\.|$)")


def get_cosim_from_steer_pairs(
    pair_i: EffectiveVector, pair_j: EffectiveVector
) -> float:
    """Compute cosine similarity directly from two low-rank LoRA differences."""

    inner = low_rank_inner(pair_i, pair_j)
    norm_i = math.sqrt(low_rank_inner(pair_i, pair_i))
    norm_j = math.sqrt(low_rank_inner(pair_j, pair_j))
    return float(np.clip(inner / (norm_i * norm_j), -1.0, 1.0))


def _layer_magnitudes(vector: EffectiveVector) -> dict[str, float]:
    by_layer = {}
    for term in vector.terms:
        match = _LAYER_NUMBER.search(term.module)
        layer = match.group(1) if match else term.module
        by_layer.setdefault(layer, []).append(term)
    result = {}
    for layer, terms in by_layer.items():
        restricted = EffectiveVector(
            vector.id, vector.behavior, vector.seed, tuple(terms)
        )
        result[layer] = math.sqrt(low_rank_inner(restricted, restricted))
    return result


def analyze(vectors: list[EffectiveVector]) -> dict:
    """Fill the symmetric cosine matrix and collect plotting metadata."""

    cosine = np.eye(len(vectors))
    for (i, pair_i), (j, pair_j) in itertools.combinations_with_replacement(
        enumerate(vectors), 2
    ):
        cosine[i, j] = cosine[j, i] = get_cosim_from_steer_pairs(pair_i, pair_j)

    return {
        "vectors": [
            {
                "id": vector.id,
                "group": vector.behavior,
                "seed": vector.seed,
                "norm": math.sqrt(low_rank_inner(vector, vector)),
                "layer_magnitudes": _layer_magnitudes(vector),
            }
            for vector in vectors
        ],
        "cosine_similarity": {
            "ids": [vector.id for vector in vectors],
            "matrix": cosine.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        default="artifacts/analysis/cosine-similarity-across-seeds/results.json",
    )
    args = parser.parse_args()

    vectors = [load_effective_vector(pair, Path.cwd()) for pair in STEER_PAIRS]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analyze(vectors), indent=2) + "\n")


if __name__ == "__main__":
    main()
