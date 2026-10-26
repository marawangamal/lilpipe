"""Regression tests for retained SEAM relearning trajectories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EVALUATOR = ROOT / "examples/tamper-resistance/scripts/evaluate_seam_figure2.py"
PLOTTER = (
    ROOT / "examples/tamper-resistance/scripts/analysis/plot_trajectory_beavertails.py"
)


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


def test_generated_base_attack_can_be_cleaned_up(monkeypatch) -> None:
    evaluator = load_evaluator()
    attack = Path(
        "artifacts/generated/hpo/llama2-7b-beavertail--base-figure2-hpo/models/"
        "llama2-7b-beavertail--base-figure2-hpo--disjoint--001"
    )
    removed = []
    monkeypatch.setattr(evaluator.shutil, "rmtree", removed.append)
    evaluator.delete_generated_attack(str(attack))
    assert removed == [attack]


def test_seam_trajectory_rows_are_numeric_and_tidy(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("seam_trajectory_plotter", PLOTTER)
    assert spec is not None and spec.loader is not None
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)

    for step in (20, 0, 10):
        output = tmp_path / "model" / f"checkpoint-{step}"
        output.mkdir(parents=True)
        (output / "summary.json").write_text("""{
              "method": "SEAM",
              "attack_data": "Disjoint",
              "learning_rate": 0.00002,
              "post": {"harmfulness": 0.25, "zero_shot_utility": 0.5}
            }""")

    rows = plotter.load_rows(tmp_path)
    harmfulness = [row for row in rows if row["Evaluation"] == "Harmfulness"]
    assert [row["Relearning step"] for row in harmfulness] == [0, 10, 20]
    assert harmfulness[0] == {
        "Relearning step": 0,
        "Score (%)": 25.0,
        "Method": "SEAM",
        "Learning rate": "2e-5",
        "Learning rate value": 2e-5,
        "Evaluation": "Harmfulness",
    }


def test_beavertails_has_main_and_learning_rate_plots() -> None:
    spec = importlib.util.spec_from_file_location("seam_trajectory_plotter", PLOTTER)
    assert spec is not None and spec.loader is not None
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    assert set(plotter.plots) == {"main", "lrs"}
    assert plotter.plots["main"]["learning_rates"] == {5e-5}
    assert plotter.plots["lrs"]["output"].endswith(
        "llama2-7b-beavertail-relearning-trajectory-ms1000--lrs.pdf"
    )


def test_learning_rate_controls_line_thickness(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    seaborn = pytest.importorskip("seaborn")
    spec = importlib.util.spec_from_file_location("seam_trajectory_plotter", PLOTTER)
    assert spec is not None and spec.loader is not None
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    captured = {}

    class Figure:
        def savefig(self, *args, **kwargs):
            pass

    class Grid:
        figure = Figure()

        def set_titles(self, *args, **kwargs):
            pass

        def set_axis_labels(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    def relplot(**kwargs):
        captured.update(kwargs)
        return Grid()

    monkeypatch.setattr(seaborn, "relplot", relplot)
    rows = [
        {
            "Method": "SEAM",
            "Learning rate": "2e-5",
            "Learning rate value": 2e-5,
            "Evaluation": "Harmfulness",
            "Relearning step": 0,
            "Score (%)": 0.0,
        }
    ]
    style = plotter.plots["lrs"]["style"]
    plotter.plot(rows, tmp_path / "plot.pdf", style)

    assert captured["size"] == "Learning rate"
    assert captured["sizes"] == [0.8, 1.4, 2.0, 2.8, 3.8]
