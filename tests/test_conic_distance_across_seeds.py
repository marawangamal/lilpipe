import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "examples" / "weight-steering" / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_script("conic_distance_gen_results")
plotter = _load_script("conic_distance_plot_results")


def _gram(*vectors):
    normalized = np.asarray(
        [np.asarray(vector, dtype=float) / np.linalg.norm(vector) for vector in vectors]
    )
    return normalized @ normalized.T


@pytest.mark.parametrize(
    ("target", "generators", "expected"),
    [
        ((1, 1), ((1, 0), (0, 1)), 0.0),
        ((-1, 1), ((1, 0), (0, 1)), 1 / np.sqrt(2)),
        ((1, 0), ((1, 0), (0, 1)), 0.0),
    ],
)
def test_cone_distance_inside_outside_and_on_boundary(target, generators, expected):
    gram = _gram(target, *generators)

    distance = generator.cone_distance_from_gram(gram, 0, (1, 2))

    assert distance == pytest.approx(expected)


def test_cone_distance_rejects_negative_coefficients():
    gram = _gram((-1, 0), (1, 0))

    assert generator.cone_distance_from_gram(gram, 0, (1,)) == pytest.approx(1.0)


def test_cone_distance_supports_redundant_singular_generators():
    gram = _gram((1, 0), (1, 0), (1, 0), (0, 1))

    assert generator.cone_distance_from_gram(gram, 0, (1, 2, 3)) == pytest.approx(0.0)


def test_complete_matrix_shape_range_and_generator_selection(monkeypatch):
    calls = []

    def record_projection(gram, target_index, generator_indices):
        calls.append((target_index, tuple(generator_indices)))
        return 0.25

    monkeypatch.setattr(generator, "cone_distance_from_gram", record_projection)
    distances = generator.compute_conic_distances(np.eye(15))

    assert distances.shape == (15, 3)
    assert np.all((0 <= distances) & (distances <= 1))
    for target_index in range(15):
        for behavior_index in range(3):
            called_target, indices = calls[target_index * 3 + behavior_index]
            expected = set(range(behavior_index * 5, behavior_index * 5 + 5))
            assert called_target == target_index
            assert set(indices) == expected


def test_own_behavior_cone_includes_target_and_has_zero_distance():
    gram = np.eye(15)

    distances = generator.compute_conic_distances(gram)

    for target_index in range(15):
        assert distances[target_index, target_index // 5] == pytest.approx(0.0)


def test_plotter_writes_expected_pdf(tmp_path):
    plotter.plot_report(np.linspace(0, 1, 45).reshape(15, 3), tmp_path)

    output = tmp_path / "conic-distance.pdf"
    assert output.exists()
    assert output.stat().st_size > 0
