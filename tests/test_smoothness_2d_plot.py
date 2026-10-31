import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/plot_smoothness_2d.py"
)


def load_script(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("plot_smoothness_2d", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_results():
    return {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {
                "label": "TAR",
                "direction_std": scale,
                "loss": [[3, 2, 3], [2, 1, 2], [3, 2, 3]],
                "trajectory": [
                    {"path": "defended", "coordinates": [0.0, 0.0]},
                    {"path": "checkpoint-16", "coordinates": [1e-8, 2e-8]},
                    {"path": "checkpoint-32", "coordinates": [2e-8, 1e-8]},
                ],
            }
            for scale in (0.03, 0.1)
        ],
    }


def test_contours_and_zoom_preserve_actual_trajectory_coordinates(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch)
    results = example_results()
    output = tmp_path / "contours.pdf"
    figure = module.plot_surfaces(results, output)
    try:
        assert output.is_file()
        panels = [axis for axis in figure.axes if axis.get_xlabel() == "x"]
        assert len(panels) == 2
        assert all(axis.name == "rectilinear" for axis in figure.axes)
        for panel, surface in zip(panels, results["models"], strict=True):
            expected = np.array(
                [point["coordinates"] for point in surface["trajectory"]]
            )
            np.testing.assert_array_equal(panel.lines[0].get_xydata(), expected)
            assert panel.collections  # Filled contours, lines, and checkpoint markers.
            assert len(panel.child_axes) == 1
            zoom = panel.child_axes[0]
            np.testing.assert_array_equal(zoom.lines[0].get_xydata(), expected)
            assert [text.get_text() for text in zoom.texts] == ["0", "16", "32"]
    finally:
        module.plt.close(figure)


def test_plotter_rejects_surfaces_without_centered_trajectories(monkeypatch, tmp_path):
    module = load_script(monkeypatch)
    results = example_results()
    del results["models"][0]["trajectory"]
    try:
        with pytest.raises(ValueError, match="centered, scale-specific trajectory"):
            module.plot_surfaces(results, tmp_path / "invalid.pdf")
    finally:
        module.plt.close("all")


def test_cli_selects_one_scale_and_keeps_its_trajectory(monkeypatch, tmp_path):
    module = load_script(monkeypatch)
    source = tmp_path / "surface.json"
    source.write_text(json.dumps(example_results()))
    output = tmp_path / "one-scale.pdf"
    captured = []

    def plot(results, destination, **kwargs):
        captured.append((results, destination))
        return module.plt.figure()

    monkeypatch.setattr(module, "plot_surfaces", plot)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), str(source), "--direction-std", "0.03", "--output", str(output)],
    )
    module.main()
    assert len(captured) == 1
    results, destination = captured[0]
    assert destination == str(output)
    assert results["models"] == example_results()["models"][:1]
