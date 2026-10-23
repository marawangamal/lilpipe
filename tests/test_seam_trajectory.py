"""Regression tests for retained SEAM relearning trajectories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EVALUATOR = ROOT / "examples/tamper-resistance/scripts/evaluate_seam_figure2.py"
PLOTTER = ROOT / "examples/tamper-resistance/scripts/analysis/plot_seam_trajectory.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("seam_figure2_evaluator", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attack_checkpoint_resolves_baseline_and_retained_steps(tmp_path: Path) -> None:
    evaluator = load_evaluator()
    assert evaluator.attack_checkpoint("baseline", "attack", None) == "attack"
    assert evaluator.attack_checkpoint("baseline", "attack", 0) == "baseline"

    checkpoint_root = tmp_path / "attack"
    checkpoint = checkpoint_root / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    assert evaluator.attack_checkpoint("baseline", str(checkpoint_root), 10) == str(
        checkpoint
    )
    with pytest.raises(ValueError, match="missing attack checkpoint"):
        evaluator.attack_checkpoint("baseline", str(checkpoint_root), 20)


def test_seam_trajectory_rows_are_numeric_and_tidy(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("seam_trajectory_plotter", PLOTTER)
    assert spec is not None and spec.loader is not None
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)

    for step in (20, 0, 10):
        output = tmp_path / "model" / f"checkpoint-{step}"
        output.mkdir(parents=True)
        (output / "summary.json").write_text("""{
              "attack_data": "Overlapping",
              "learning_rate": 0.00002,
              "post": {"harmfulness": 0.25, "zero_shot_utility": 0.5}
            }""")

    rows = plotter.load_rows(tmp_path)
    harmfulness = [row for row in rows if row["Metric"] == "Harmfulness"]
    assert [row["Relearning step"] for row in harmfulness] == [0, 10, 20]
    assert harmfulness[0] == {
        "Relearning step": 0,
        "Score (%)": 25.0,
        "Metric": "Harmfulness",
        "Learning rate": "2e-5",
        "Attack data": "Overlapping",
    }
