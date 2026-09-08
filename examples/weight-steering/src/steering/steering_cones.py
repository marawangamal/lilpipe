#!/usr/bin/env python3
"""Measure contrastive LoRA directions against within- and cross-behavior cones."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
import yaml


class ConeAnalysisError(ValueError):
    """Raised when adapters or the analysis configuration are invalid."""


@dataclass(frozen=True)
class FactorTerm:
    """One scaled low-rank matrix in an effective update."""

    module: str
    a: torch.Tensor
    b: torch.Tensor
    scale: float


@dataclass(frozen=True)
class EffectiveVector:
    """A named sum of low-rank module updates."""

    id: str
    behavior: str
    seed: int
    terms: tuple[FactorTerm, ...]


_LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")


def low_rank_inner(left: EffectiveVector, right: EffectiveVector) -> float:
    """Return a Frobenius inner product without forming dense BA matrices."""

    total = 0.0
    right_by_module: dict[str, list[FactorTerm]] = {}
    for term in right.terms:
        right_by_module.setdefault(term.module, []).append(term)
    for first in left.terms:
        for second in right_by_module.get(first.module, ()):
            if first.a.shape[1] != second.a.shape[1] or first.b.shape[0] != second.b.shape[0]:
                raise ConeAnalysisError(
                    f"LoRA module {first.module!r} has incompatible base dimensions"
                )
            bt_b = first.b.T @ second.b
            a_at = first.a @ second.a.T
            total += first.scale * second.scale * torch.sum(bt_b * a_at).item()
    return float(total)


def gram_matrix(vectors: list[EffectiveVector]) -> np.ndarray:
    """Construct a symmetric Gram matrix from low-rank factors."""

    result = np.empty((len(vectors), len(vectors)), dtype=np.float64)
    for i, left in enumerate(vectors):
        for j in range(i + 1):
            value = low_rank_inner(left, vectors[j])
            result[i, j] = result[j, i] = value
    return result


def project_onto_cone(
    source_gram: np.ndarray, target_inner: np.ndarray, target_norm_sq: float = 1.0
) -> tuple[np.ndarray, float]:
    """Solve a small nonnegative least-squares problem from Gram quantities.

    All active sets are considered. Experiment cones contain only four or five
    sources, making this exact approach fast and robust to singular Gram matrices.
    """

    source_gram = np.asarray(source_gram, dtype=np.float64)
    target_inner = np.asarray(target_inner, dtype=np.float64)
    count = len(target_inner)
    if source_gram.shape != (count, count) or target_norm_sq < 0:
        raise ConeAnalysisError("Invalid dimensions or norm for cone projection")
    best_coefficients = np.zeros(count, dtype=np.float64)
    best_squared = float(target_norm_sq)
    for mask in range(1, 1 << count):
        active = np.array([index for index in range(count) if mask & (1 << index)])
        sub_gram = source_gram[np.ix_(active, active)]
        sub_inner = target_inner[active]
        coefficients, *_ = np.linalg.lstsq(sub_gram, sub_inner, rcond=None)
        if np.any(coefficients < -1e-10):
            continue
        coefficients = np.maximum(coefficients, 0.0)
        squared = float(
            target_norm_sq
            - 2.0 * coefficients @ sub_inner
            + coefficients @ sub_gram @ coefficients
        )
        if squared < best_squared:
            best_squared = squared
            best_coefficients = np.zeros(count, dtype=np.float64)
            best_coefficients[active] = coefficients
    return best_coefficients, math.sqrt(max(0.0, best_squared))


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = path / "adapter_model.safetensors"
    binary_path = path / "adapter_model.bin"
    if safetensors_path.is_file():
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ConeAnalysisError("safetensors is required to read LoRA adapters") from error
        return load_file(str(safetensors_path), device="cpu")
    if binary_path.is_file():
        state = torch.load(binary_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ConeAnalysisError(f"Adapter state at {binary_path} is not a mapping")
        return state
    raise ConeAnalysisError(f"No adapter_model.safetensors or adapter_model.bin in {path}")


def _adapter_terms(path: Path, sign: float) -> tuple[FactorTerm, ...]:
    config_path = path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConeAnalysisError(f"Could not read adapter config {config_path}: {error}") from error
    if config.get("peft_type") != "LORA":
        raise ConeAnalysisError(f"Adapter {path} is not a standard LoRA adapter")
    unsupported = [
        name for name in ("use_rslora", "use_dora") if config.get(name, False)
    ]
    if unsupported or config.get("modules_to_save"):
        raise ConeAnalysisError(f"Adapter {path} uses unsupported options: {unsupported or ['modules_to_save']}")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)):
        raise ConeAnalysisError(f"Adapter {path} has invalid r or lora_alpha")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    state = _load_state(path)
    for key, tensor in state.items():
        match = _LORA_KEY.match(key)
        if match is None:
            raise ConeAnalysisError(f"Unsupported saved adapter tensor {key!r} in {path}")
        pairs.setdefault(match.group("module"), {})[match.group("side")] = tensor.detach().to(torch.float64)
    if not pairs:
        raise ConeAnalysisError(f"Adapter {path} has no LoRA tensors")
    terms = []
    for module, sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            raise ConeAnalysisError(f"LoRA module {module!r} in {path} is missing A or B")
        a, b = sides["A"], sides["B"]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ConeAnalysisError(f"LoRA module {module!r} in {path} has invalid tensor shapes")
        module_rank = a.shape[0]
        rank_pattern = config.get("rank_pattern") or {}
        alpha_pattern = config.get("alpha_pattern") or {}
        configured_rank = rank_pattern.get(module, rank)
        module_alpha = alpha_pattern.get(module, alpha)
        if configured_rank != module_rank:
            raise ConeAnalysisError(f"LoRA module {module!r} rank disagrees with adapter config")
        terms.append(FactorTerm(module, a, b, sign * float(module_alpha) / module_rank))
    return tuple(terms)


def load_effective_vector(entry: dict[str, Any], root: Path) -> EffectiveVector:
    """Load positive-minus-negative LoRA factors for one configured vector."""

    required = {"id", "behavior", "seed", "positive_adapter", "negative_adapter"}
    if set(entry) != required:
        raise ConeAnalysisError(f"Vector entry must contain exactly {sorted(required)}")
    if not isinstance(entry["id"], str) or not entry["id"] or not isinstance(entry["behavior"], str) or not entry["behavior"]:
        raise ConeAnalysisError("Vector id and behavior must be non-empty strings")
    if type(entry["seed"]) is not int:
        raise ConeAnalysisError("Vector seed must be an integer")
    positive = root / entry["positive_adapter"]
    negative = root / entry["negative_adapter"]
    positive_terms = _adapter_terms(positive, 1.0)
    negative_terms = _adapter_terms(negative, -1.0)
    if {term.module for term in positive_terms} != {term.module for term in negative_terms}:
        raise ConeAnalysisError(f"Vector {entry['id']!r} has unmatched positive/negative LoRA modules")
    return EffectiveVector(entry["id"], entry["behavior"], entry["seed"], positive_terms + negative_terms)


def load_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate an analysis configuration."""

    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConeAnalysisError(f"Could not read analysis config {path}: {error}") from error
    required = {"tolerance", "output_json", "output_png", "vectors"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ConeAnalysisError(f"Analysis config must contain exactly {sorted(required)}")
    if not isinstance(raw["tolerance"], (int, float)) or raw["tolerance"] < 0:
        raise ConeAnalysisError("tolerance must be a nonnegative number")
    if not all(isinstance(raw[key], str) and raw[key] for key in ("output_json", "output_png")):
        raise ConeAnalysisError("Output paths must be non-empty strings")
    if not isinstance(raw["vectors"], list) or len(raw["vectors"]) < 2 or not all(isinstance(item, dict) for item in raw["vectors"]):
        raise ConeAnalysisError("vectors must be a list of at least two mappings")
    ids = [item.get("id") for item in raw["vectors"]]
    if len(ids) != len(set(ids)):
        raise ConeAnalysisError("Vector ids must be unique")
    return raw


def analyze(vectors: list[EffectiveVector], tolerance: float) -> dict[str, Any]:
    """Compute norms, cosines, and cross/within cone projections."""

    gram = gram_matrix(vectors)
    squared_norms = np.diag(gram)
    if np.any(squared_norms <= 0):
        bad = [vectors[i].id for i in np.flatnonzero(squared_norms <= 0)]
        raise ConeAnalysisError(f"Zero-norm effective vectors: {bad}")
    norms = np.sqrt(squared_norms)
    cosine = gram / np.outer(norms, norms)
    cosine = np.clip(cosine, -1.0, 1.0)
    projections = []
    for target_index, target in enumerate(vectors):
        for kind in ("cross", "within"):
            indices = [
                index for index, source in enumerate(vectors)
                if index != target_index
                and ((source.behavior != target.behavior) if kind == "cross" else (source.behavior == target.behavior))
            ]
            if not indices:
                raise ConeAnalysisError(f"Vector {target.id!r} has no {kind}-behavior cone sources")
            source_gram = cosine[np.ix_(indices, indices)]
            target_inner = cosine[indices, target_index]
            coefficients, distance = project_onto_cone(source_gram, target_inner)
            nearest = math.sqrt(max(0.0, 1.0 - max(0.0, float(np.max(target_inner))) ** 2))
            projections.append({
                "target": target.id,
                "kind": kind,
                "sources": [vectors[index].id for index in indices],
                "coefficients": coefficients.tolist(),
                "relative_distance": distance,
                "nearest_source_distance": nearest,
                "combination_improvement": nearest - distance,
                "is_member": distance <= tolerance,
            })
    return {
        "tolerance": tolerance,
        "vectors": [{"id": vector.id, "behavior": vector.behavior, "seed": vector.seed, "norm": float(norms[index])} for index, vector in enumerate(vectors)],
        "cosine_similarity": {"ids": [vector.id for vector in vectors], "matrix": cosine.tolist()},
        "projections": projections,
    }


def plot_report(report: dict[str, Any], output: Path) -> None:
    """Write the cosine heatmap and cross/within distance summary."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = report["cosine_similarity"]["ids"]
    matrix = np.asarray(report["cosine_similarity"]["matrix"])
    by_kind = {kind: {item["target"]: item["relative_distance"] for item in report["projections"] if item["kind"] == kind} for kind in ("cross", "within")}
    figure, (heatmap_axis, distance_axis) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.35, 1]})
    image = heatmap_axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    heatmap_axis.set_xticks(range(len(ids)), ids, rotation=60, ha="right", fontsize=8)
    heatmap_axis.set_yticks(range(len(ids)), ids, fontsize=8)
    heatmap_axis.set_title("Effective-vector cosine similarity")
    figure.colorbar(image, ax=heatmap_axis, fraction=0.046)
    positions = np.arange(len(ids))
    width = 0.38
    distance_axis.bar(positions - width / 2, [by_kind["cross"][item] for item in ids], width, label="cross")
    distance_axis.bar(positions + width / 2, [by_kind["within"][item] for item in ids], width, label="within")
    distance_axis.axhline(report["tolerance"], color="black", linestyle="--", linewidth=1, label="membership tolerance")
    distance_axis.set_xticks(positions, ids, rotation=60, ha="right", fontsize=8)
    distance_axis.set_ylabel("Relative cone distance")
    distance_axis.set_title("Cone projection residuals")
    distance_axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    root = Path.cwd()
    vectors = [load_effective_vector(entry, root) for entry in config["vectors"]]
    report = analyze(vectors, float(config["tolerance"]))
    json_path = root / config["output_json"]
    png_path = root / config["output_png"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, png_path)


if __name__ == "__main__":
    main()
