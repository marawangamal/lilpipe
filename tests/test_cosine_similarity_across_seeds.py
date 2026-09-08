import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

import lilpipe


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering"


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


steering_cones = _load_module("steering_cones", "src/steering/steering_cones.py")
control = _load_module(
    "compute_cosine_similarity_gen_results",
    "scripts/analysis/compute_cosine_similarity_gen_results.py",
)


def _normalized_training_config(path):
    config = yaml.safe_load(path.read_text())
    config["datasets"] = "BEHAVIOR_DATASET"
    config["dataset_prepared_path"] = "PREPARED_PATH"
    config["output_dir"] = "OUTPUT_PATH"
    return config


def test_control_training_configs_match_ns_hyperparameters():
    training = EXAMPLE / "configs" / "training"
    expected = _normalized_training_config(training / "non-sycophantic.yml")

    assert _normalized_training_config(training / "honest.yml") == expected
    assert _normalized_training_config(training / "dishonest.yml") == expected


def test_control_pipeline_has_thirty_unique_arms(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(
        "configs/experiments/pipeline-cosine-similarity-across-seeds.yml"
    ).plan()
    training = [stage for stage in plan.stages if stage.id.startswith("train-seeded-")]

    assert len(plan.stages) == 30
    assert len(training) == 30
    assert len({stage.id for stage in training}) == 30
    assert len({stage.args[2] for stage in training}) == 30
    assert len({stage.args[3] for stage in training}) == 30
    for stage in training:
        seed = stage.id.rsplit("-", 1)[1]
        assert seed in {"42", "43", "44", "45", "46"}
        assert stage.args[1] == seed
        assert stage.args[2].endswith(f"-{seed}")
        assert stage.args[3].endswith(f"-{seed}")
        assert stage.script == "scripts/slurm/train.sbatch"
        assert stage.depends_on == ()


def _term(module, value):
    return steering_cones.FactorTerm(
        module,
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([[value]], dtype=torch.float64),
        1.0,
    )


def _vector(name, group, seed, x, y):
    return steering_cones.EffectiveVector(
        name, group, seed, (_term("model.layers.0.proj", x), _term("model.layers.1.proj", y))
    )


def test_cosine_grouping_and_layer_aggregation_with_low_rank_vectors():
    vectors = []
    directions = {"Honesty": (1.0, 0.0), "NS": (0.0, 1.0), "NC": (1.0, 1.0)}
    for group, (x, y) in directions.items():
        for seed in range(42, 47):
            vectors.append(_vector(f"{group}-{seed}", group, seed, x, y))

    report = control.analyze(vectors)

    assert np.asarray(report["cosine_similarity"]["matrix"]).shape == (15, 15)
    matrix = np.asarray(report["cosine_similarity"]["matrix"])
    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[0, 5] == pytest.approx(0.0)
    assert matrix[0, 10] == pytest.approx(2 ** -0.5)
    honesty = report["vectors"][0]
    assert honesty["layer_magnitudes"] == {"0": 1.0, "1": 0.0}


def test_registry_declares_three_seeded_groups():
    registry_path = EXAMPLE / "configs" / "registries" / "models.yml"
    models = yaml.safe_load(registry_path.read_text())["models"]

    assert len(control.STEER_PAIRS) == 15
    for group in ("Honesty", "NS", "NC"):
        assert sorted(
            item["seed"]
            for item in control.STEER_PAIRS
            if item["behavior"] == group
        ) == list(range(42, 47))

    seeded = {model_id: model for model_id, model in models.items() if "-Seed-" in model_id}
    assert len(seeded) == 30
    assert len({model["artifact"] for model in seeded.values()}) == 30
    expected_configs = {
        "Honest": "honest.yml",
        "Dishonest": "dishonest.yml",
        "Non-Sycophantic": "non-sycophantic.yml",
        "Sycophantic": "sycophantic.yml",
        "Non-Cheat": "non-cheat.yml",
        "Cheat": "cheat.yml",
    }
    for model_id, model in seeded.items():
        behavior, seed = model_id.removeprefix("SmolLM3-3B-HMO-FT-").rsplit("-Seed-", 1)
        producer = model["producer"]
        assert producer["script"] == "scripts/slurm/train.sbatch"
        assert producer["args"] == [
            f"configs/training/{expected_configs[behavior]}",
            seed,
            model["artifact"],
            f"artifacts/data/cosine-similarity-across-seeds/{behavior.lower()}-{seed}",
        ]

    assert [item["positive_adapter"].split("/")[-1].rsplit("-", 1)[0] for item in control.STEER_PAIRS] == (
        ["honest"] * 5 + ["non-sycophantic"] * 5 + ["non-cheat"] * 5
    )
    assert [item["negative_adapter"].split("/")[-1].rsplit("-", 1)[0] for item in control.STEER_PAIRS] == (
        ["dishonest"] * 5 + ["sycophantic"] * 5 + ["cheat"] * 5
    )
