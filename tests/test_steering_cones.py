import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

import lilpipe


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering"
MODULE_PATH = EXAMPLE / "src" / "steering" / "steering_cones.py"
SPEC = importlib.util.spec_from_file_location("steering_cones", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
steering_cones = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = steering_cones
SPEC.loader.exec_module(steering_cones)


def _vector(name, terms):
    return steering_cones.EffectiveVector(name, "behavior", 42, tuple(terms))


def _term(module, a, b, scale):
    return steering_cones.FactorTerm(
        module,
        torch.tensor(a, dtype=torch.float64),
        torch.tensor(b, dtype=torch.float64),
        scale,
    )


def _dense(vector):
    modules = {}
    for term in vector.terms:
        update = term.scale * (term.b @ term.a)
        modules[term.module] = modules.get(term.module, torch.zeros_like(update)) + update
    return modules


def _explicit_inner(left, right):
    left_dense = _dense(left)
    right_dense = _dense(right)
    return sum(torch.sum(value * right_dense[module]).item() for module, value in left_dense.items() if module in right_dense)


def test_low_rank_inner_matches_materialized_contrastive_updates():
    left = _vector("left", [
        _term("one", [[1, 2, 0], [0, 1, 1]], [[1, 0], [2, 1]], 2.0),
        _term("one", [[2, -1, 1]], [[1], [3]], -0.5),
        _term("left-only", [[1, 0]], [[2]], 3.0),
    ])
    right = _vector("right", [
        _term("one", [[1, 0, 2]], [[-1], [2]], 1.5),
        _term("right-only", [[2]], [[1]], 4.0),
    ])

    assert steering_cones.low_rank_inner(left, right) == pytest.approx(
        _explicit_inner(left, right)
    )
    gram = steering_cones.gram_matrix([left, right])
    assert gram[0, 1] == pytest.approx(gram[1, 0])
    assert gram[0, 0] == pytest.approx(_explicit_inner(left, left))


@pytest.mark.parametrize(
    ("gram", "inner", "expected"),
    [
        ([[1, 0], [0, 1]], [0.6, 0.8], 0.0),  # inside
        ([[1, 0], [0, 1]], [1.0, 0.0], 0.0),  # boundary
        ([[1]], [-1.0], 1.0),  # outside
        ([[1, 1], [1, 1]], [1.0, 1.0], 0.0),  # redundant/singular
    ],
)
def test_cone_projection_known_cases(gram, inner, expected):
    coefficients, distance = steering_cones.project_onto_cone(
        np.asarray(gram), np.asarray(inner)
    )

    assert np.all(coefficients >= 0)
    assert distance == pytest.approx(expected, abs=1e-8)


def _write_adapter(path, modules=None, **config_overrides):
    path.mkdir()
    config = {"peft_type": "LORA", "r": 1, "lora_alpha": 2, **config_overrides}
    (path / "adapter_config.json").write_text(json.dumps(config))
    if modules is None:
        modules = {"layer": (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]]))}
    state = {}
    for module, (a, b) in modules.items():
        if a is not None:
            state[f"base.{module}.lora_A.weight"] = a
        if b is not None:
            state[f"base.{module}.lora_B.weight"] = b
    torch.save(state, path / "adapter_model.bin")


def test_load_effective_vector_scales_and_subtracts_adapters(tmp_path):
    _write_adapter(tmp_path / "positive")
    _write_adapter(
        tmp_path / "negative",
        {"layer": (torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0]]))},
    )
    entry = {"id": "v", "behavior": "b", "seed": 42, "positive_adapter": "positive", "negative_adapter": "negative"}

    vector = steering_cones.load_effective_vector(entry, tmp_path)
    dense = _dense(vector)["base.layer"]

    assert torch.equal(dense, torch.tensor([[2.0, 12.0]], dtype=torch.float64))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda path: _write_adapter(path, {"layer": (torch.tensor([[1.0]]), None)}), "missing A or B"),
        (lambda path: _write_adapter(path, use_rslora=True), "unsupported options"),
    ],
)
def test_adapter_validation(tmp_path, mutation, match):
    mutation(tmp_path / "bad")
    with pytest.raises(steering_cones.ConeAnalysisError, match=match):
        steering_cones._adapter_terms(tmp_path / "bad", 1.0)


def test_rejects_unmatched_modules_and_zero_norm(tmp_path):
    _write_adapter(tmp_path / "positive")
    _write_adapter(tmp_path / "negative", {"other": (torch.tensor([[1.0]]), torch.tensor([[1.0]]))})
    entry = {"id": "v", "behavior": "b", "seed": 42, "positive_adapter": "positive", "negative_adapter": "negative"}
    with pytest.raises(steering_cones.ConeAnalysisError, match="unmatched"):
        steering_cones.load_effective_vector(entry, tmp_path)

    zero = _vector("zero", [_term("m", [[0.0]], [[1.0]], 1.0)])
    other = steering_cones.EffectiveVector("other", "other", 43, (_term("m", [[1.0]], [[1.0]], 1.0),))
    with pytest.raises(steering_cones.ConeAnalysisError, match="Zero-norm"):
        steering_cones.analyze([zero, other], 1e-5)


def test_malformed_analysis_config_is_rejected(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"tolerance": -1, "vectors": []}))
    with pytest.raises(steering_cones.ConeAnalysisError, match="exactly"):
        steering_cones.load_config(path)


def test_seeded_cone_pipeline_has_twenty_unique_arms_and_one_analysis(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/steering-cones.yml").plan()
    training = [stage for stage in plan.stages if stage.id.startswith("train-cone-")]

    assert len(plan.stages) == 21
    assert len(training) == 20
    assert len({stage.id for stage in training}) == 20
    for stage in training:
        seed = stage.id.rsplit("-", 1)[1]
        assert stage.args[1] == seed
        assert stage.args[2].endswith(f"-{seed}")
        assert stage.args[3].endswith(f"-{seed}")
        assert stage.depends_on == ()
    analysis = plan.stage_index["analyze-steering-cones"]
    assert set(analysis.depends_on) == {stage.id for stage in training}
    assert analysis.args == ("configs/analysis/steering-cones.yml",)


def test_training_script_accepts_seed_output_and_cache_overrides():
    script = (EXAMPLE / "scripts" / "slurm" / "train.sbatch").read_text()

    assert 'overrides+=(--seed "$seed")' in script
    assert 'overrides+=(--output-dir "$output_dir")' in script
    assert 'overrides+=(--dataset-prepared-path "$dataset_prepared_path")' in script
