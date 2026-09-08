#!/usr/bin/env python3
"""Plot cosine similarities between the seeded effective steering vectors."""

import itertools as it
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from steering.steering_cones import load_effective_vector, low_rank_inner  # noqa: E402


config = yaml.safe_load(
    (PROJECT / "configs/analysis/steering-cones.yml").read_text()
)
model_name_or_path_lst = config["vectors"]
vectors = [load_effective_vector(spec, PROJECT) for spec in model_name_or_path_lst]

n = len(vectors)
cosine_sim = torch.zeros(n, n, dtype=torch.float64)
norms = torch.tensor(
    [low_rank_inner(vector, vector) ** 0.5 for vector in vectors]
)
for (i, model_i), (j, model_j) in it.product(
    enumerate(vectors), enumerate(vectors)
):
    cosine_sim[i, j] = low_rank_inner(model_i, model_j) / (norms[i] * norms[j])

labels = [vector.id.replace("non-cheating", "NC").replace("non-sycophancy", "NS") for vector in vectors]
off_diagonal = ~torch.eye(n, dtype=torch.bool)
limit = cosine_sim[off_diagonal].abs().max().item()

figure, axis = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cosine_sim.numpy(),
    mask=torch.eye(n, dtype=torch.bool).numpy(),
    xticklabels=labels,
    yticklabels=labels,
    cmap="vlag",
    center=0,
    vmin=-limit,
    vmax=limit,
    annot=True,
    fmt=".3f",
    square=True,
    ax=axis,
)
axis.set_title("Seeded effective steering-vector cosine similarity")
axis.tick_params(axis="x", rotation=45)
axis.tick_params(axis="y", rotation=0)
figure.tight_layout()

output = PROJECT / "scripts/analysis/plots/steering-vector-cosines.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, bbox_inches="tight")
plt.close(figure)
print(output)
