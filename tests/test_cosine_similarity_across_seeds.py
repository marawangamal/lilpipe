from pathlib import Path
import re

import pytest
import yaml

import lilpipe


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering"


def _normalized_training_config(path):
    config = yaml.safe_load(path.read_text())
    config["datasets"] = "BEHAVIOR_DATASET"
    config["dataset_prepared_path"] = "PREPARED_PATH"
    config["output_dir"] = "OUTPUT_PATH"
    return config


def test_control_training_configs_match_ns_hyperparameters():
    training = EXAMPLE / "configs" / "training" / "smollm3"
    expected = _normalized_training_config(training / "cfierro-non-sycophantic.yml")

    assert _normalized_training_config(training / "tara-honest.yml") == expected
    assert _normalized_training_config(training / "tara-dishonest.yml") == expected


def test_control_pipeline_has_thirty_unique_arms(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(
        "configs/experiments/smollm3/cosine-similarity-across-seeds.yml"
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


def test_registry_declares_three_seeded_groups():
    registry_path = EXAMPLE / "configs" / "registries" / "smollm3" / "models.yml"
    models = yaml.safe_load(registry_path.read_text())["models"]

    seeded = {
        model_id: model
        for model_id, model in models.items()
        if re.search(r"-Seed-\d+$", model_id)
    }
    assert len(seeded) == 30
    assert len({model["artifact"] for model in seeded.values()}) == 30
    expected_configs = {
        "Honest": "tara-honest.yml",
        "Dishonest": "tara-dishonest.yml",
        "Non-Sycophantic": "cfierro-non-sycophantic.yml",
        "Sycophantic": "cfierro-sycophantic.yml",
        "Non-Cheat": "mbpp-non-cheat.yml",
        "Cheat": "mbpp-cheat.yml",
    }
    for model_id, model in seeded.items():
        behavior, seed = model_id.removeprefix("SmolLM3-3B-HMO-FT-").rsplit("-Seed-", 1)
        producer = model["producer"]
        assert producer["script"] == "scripts/slurm/train.sbatch"
        assert producer["args"] == [
            f"configs/training/smollm3/{expected_configs[behavior]}",
            seed,
            model["artifact"],
            f"artifacts/data/cosine-similarity-across-seeds/{behavior.lower()}-{seed}",
        ]
