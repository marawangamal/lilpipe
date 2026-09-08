#!/usr/bin/env python3
import itertools
import json
from functools import cache
from pathlib import Path
import re

import numpy as np
from safetensors.torch import load_file
import torch


SEEDS = range(42, 47)


@cache
def load_steer_pair(pair):
    terms = []
    for sign, path in zip((1, -1), pair):
        config = json.loads((path / "adapter_config.json").read_text())
        scale = sign * config["lora_alpha"] / config["r"]
        factors = {}
        for key, tensor in load_file(str(path / "adapter_model.safetensors")).items():
            module, side = re.match(r"^(.+)\.lora_([AB])(?:\.[^.]+)?\.weight$", key).groups()
            factors.setdefault(module, {})[side] = tensor.double()
        terms += [(module, parts["A"], parts["B"], scale) for module, parts in factors.items()]
    return terms


def dot(pair_i, pair_j):
    return sum(
        scale_i * scale_j * torch.sum((b_i.T @ b_j) * (a_i @ a_j.T))
        for module_i, a_i, b_i, scale_i in load_steer_pair(pair_i)
        for module_j, a_j, b_j, scale_j in load_steer_pair(pair_j)
        if module_i == module_j
    ).item()


def get_cosim_from_steer_pairs(pair_i, pair_j):
    return dot(pair_i, pair_j) / np.sqrt(dot(pair_i, pair_i) * dot(pair_j, pair_j))


if __name__ == "__main__":
    root = Path("artifacts/models/cosine-similarity-across-seeds")
    steer_name_or_path_pairs = [
        (root / f"{positive}-{seed}", root / f"{negative}-{seed}")
        for positive, negative in [
            ("honest", "dishonest"),
            ("non-sycophantic", "sycophantic"),
            ("non-cheat", "cheat"),
        ]
        for seed in SEEDS
    ]

    cosim = np.eye(len(steer_name_or_path_pairs))
    for (i, pair_i), (j, pair_j) in itertools.combinations(
        enumerate(steer_name_or_path_pairs), 2
    ):
        cosim[i, j] = cosim[j, i] = get_cosim_from_steer_pairs(pair_i, pair_j)

    output = Path("artifacts/analysis/cosine-similarity-across-seeds/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cosim.tolist(), indent=2) + "\n")
