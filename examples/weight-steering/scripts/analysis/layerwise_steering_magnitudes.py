#!/usr/bin/env python3
"""Plot per-layer magnitudes of the seeded effective steering vectors."""

from collections import defaultdict
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from steering.steering_cones import (  # noqa: E402
    EffectiveVector,
    load_effective_vector,
    low_rank_inner,
)


config = yaml.safe_load(
    (PROJECT / "configs/analysis/steering-cones.yml").read_text()
)
vectors = [load_effective_vector(spec, PROJECT) for spec in config["vectors"]]
layer_pattern = re.compile(r"\.layers\.(\d+)\.")

magnitudes = defaultdict(dict)
for vector in vectors:
    layer_terms = defaultdict(list)
    for term in vector.terms:
        match = layer_pattern.search(term.module)
        if match is None:
            raise ValueError(f"Could not identify layer in {term.module!r}")
        layer_terms[int(match.group(1))].append(term)
    for layer, terms in layer_terms.items():
        layer_vector = EffectiveVector(
            vector.id, vector.behavior, vector.seed, tuple(terms)
        )
        magnitudes[vector.behavior][vector.seed, layer] = low_rank_inner(
            layer_vector, layer_vector
        ) ** 0.5

layers = sorted({layer for values in magnitudes.values() for _, layer in values})
figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for axis, (behavior, short_name) in zip(
    axes, (("non-cheating", "NC"), ("non-sycophancy", "NS"))
):
    seed_curves = []
    for seed in range(42, 47):
        curve = torch.tensor(
            [magnitudes[behavior][seed, layer] for layer in layers]
        )
        seed_curves.append(curve)
        axis.plot(layers, curve, alpha=0.55, linewidth=1.4, label=f"seed {seed}")
    mean = torch.stack(seed_curves).mean(dim=0)
    axis.plot(layers, mean, color="black", linewidth=2.3, label="mean")
    axis.set_title(f"{short_name}: {behavior}")
    axis.set_xlabel("Transformer layer")
    axis.grid(alpha=0.2)
axes[0].set_ylabel(r"Layer $\|\Delta W_{+}-\Delta W_{-}\|_F$")
axes[1].legend(frameon=False, ncol=2)
figure.suptitle("Layer-wise effective steering-vector magnitude")
figure.tight_layout()

output = PROJECT / "scripts/analysis/plots/layerwise-steering-magnitudes.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, bbox_inches="tight")
plt.close(figure)
print(output)
