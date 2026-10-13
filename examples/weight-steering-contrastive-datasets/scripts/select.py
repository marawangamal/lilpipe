#!/usr/bin/env python3
"""Promote the best completed proxy candidate in each comparison cell."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

import yaml


def main() -> None:
    root = Path("artifacts/generated/hpo")
    groups = yaml.safe_load((root / "groups.yml").read_text())
    model_registry = yaml.safe_load((root / "models.yml").read_text())["models"]
    baseline = json.loads((root / "proxy/baseline-target.json").read_text())
    oracle = json.loads((root / "proxy/baseline-oracle.json").read_text())
    target_forget = float(baseline["forget10_probability"])
    oracle_forget = float(oracle["forget10_probability"])
    target_retain = float(baseline["retain90_probability"])
    selected = {}
    for group, candidates in groups.items():
        scored = []
        for candidate in candidates:
            path = root / "proxy" / f"{candidate}.json"
            if path.is_file():
                metrics = json.loads(path.read_text())
                forget_progress = max(
                    0.0,
                    min(
                        1.0,
                        (target_forget - float(metrics["forget10_probability"]))
                        / (target_forget - oracle_forget),
                    ),
                )
                retain_utility = max(
                    0.0,
                    min(
                        1.0,
                        float(metrics["retain90_probability"]) / target_retain,
                    ),
                )
                unconstrained_score = (
                    2
                    * forget_progress
                    * retain_utility
                    / (forget_progress + retain_utility)
                    if forget_progress + retain_utility
                    else 0.0
                )
                # TOFU Model Utility is a harmonic mean and collapses when any
                # retained capability is destroyed. Enforce a conservative
                # likelihood floor before comparing forgetting tradeoffs.
                score = unconstrained_score if retain_utility >= 0.8 else 0.0
                if isfinite(score):
                    metrics["forget_progress"] = forget_progress
                    metrics["retain_utility"] = retain_utility
                    metrics["selection_score"] = score
                    scored.append((score, candidate, metrics))
        if not scored:
            raise ValueError(f"no completed candidates for {group}")
        score, candidate, metrics = max(scored)
        producer_args = model_registry[candidate]["producer"]["args"]
        hyperparameters = {
            producer_args[index].removeprefix("--"): producer_args[index + 1]
            for index in range(0, len(producer_args) - 1)
            if producer_args[index]
            in {"--alpha", "--learning-rate", "--beta", "--steps"}
        }
        selected[group] = {
            "candidate": candidate,
            "model": str(root / "models" / candidate),
            "hyperparameters": hyperparameters,
            "selection_score": score,
            "proxy": metrics,
        }
    output = Path("configs/selected.yml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(selected, sort_keys=False))
    models = {
        cell: {
            "base_model_name_or_path": value["model"],
            "cell": cell,
        }
        for cell, value in selected.items()
    }
    Path("configs/selected-models.yml").write_text(
        yaml.safe_dump({"models": models}, sort_keys=False)
    )
    experiment_path = Path("configs/experiments/selected.yml")
    experiment = yaml.safe_load(experiment_path.read_text())
    experiment["models"] = list(selected)
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))
    print(output)


if __name__ == "__main__":
    main()
