#!/usr/bin/env python3
"""Compute cosine similarities between seeded pairs of LoRA adapters."""

import itertools
import json
from functools import cache
from pathlib import Path
import re

import numpy as np
from safetensors.torch import load_file
import torch


SEEDS = range(42, 47)
ROOT = Path("artifacts/models/cosine-similarity-across-seeds")
BEHAVIORS = (
    ("Honesty", "honest", "dishonest"),
    ("NS", "non-sycophantic", "sycophantic"),
    ("NC", "non-cheat", "cheat"),
)
STEER_NAME_OR_PATH_PAIRS = [
    (
        f"{behavior.lower()}-{seed}",
        ROOT / f"{positive}-{seed}",
        ROOT / f"{negative}-{seed}",
    )
    for behavior, positive, negative in BEHAVIORS
    for seed in SEEDS
]
OUTPUT = Path("artifacts/analysis/cosine-similarity-across-seeds/results.json")
LORA_KEY = re.compile(r"^(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$")


@cache
def load_adapter(path):
    config = json.loads((path / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    factors = {}
    for key, tensor in load_file(str(path / "adapter_model.safetensors")).items():
        module, side = LORA_KEY.match(key).groups()
        factors.setdefault(module, {})[side] = tensor.double()
    return {module: (parts["A"], parts["B"], scale) for module, parts in factors.items()}


def adapter_inner(adapter_i, adapter_j):
    total = 0.0
    for module, (a_i, b_i, scale_i) in adapter_i.items():
        a_j, b_j, scale_j = adapter_j[module]
        total += scale_i * scale_j * torch.sum((b_i.T @ b_j) * (a_i @ a_j.T))
    return total.item()


def direction_inner(pair_i, pair_j):
    _, positive_i, negative_i = pair_i
    _, positive_j, negative_j = pair_j
    positive_i, negative_i = load_adapter(positive_i), load_adapter(negative_i)
    positive_j, negative_j = load_adapter(positive_j), load_adapter(negative_j)
    return (
        adapter_inner(positive_i, positive_j)
        - adapter_inner(positive_i, negative_j)
        - adapter_inner(negative_i, positive_j)
        + adapter_inner(negative_i, negative_j)
    )


def get_cosim_from_steer_pairs(pair_i, pair_j):
    return direction_inner(pair_i, pair_j) / np.sqrt(
        direction_inner(pair_i, pair_i) * direction_inner(pair_j, pair_j)
    )


cosim = np.eye(len(STEER_NAME_OR_PATH_PAIRS))
for (i, pair_i), (j, pair_j) in itertools.combinations_with_replacement(
    enumerate(STEER_NAME_OR_PATH_PAIRS), 2
):
    cosim[i, j] = cosim[j, i] = get_cosim_from_steer_pairs(pair_i, pair_j)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(
        {
            "ids": [pair[0] for pair in STEER_NAME_OR_PATH_PAIRS],
            "matrix": cosim.tolist(),
        },
        indent=2,
    )
    + "\n"
)
