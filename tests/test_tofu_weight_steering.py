import importlib.util
from pathlib import Path

import pytest
import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering"


def _load_metrics():
    path = EXAMPLE / "scripts" / "evaluation" / "tofu.py"
    spec = importlib.util.spec_from_file_location("tofu_metrics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tofu_dag_has_controls_arms_and_three_alpha_builds(monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/tofu/forget05.yml").plan()

    assert len(plan.stages) == 14
    oracle = "train-SmolLM3-3B-TOFU-Retain95-Oracle"
    for alpha in ("0.5", "1", "2"):
        build = plan.stage_index[f"build-SmolLM3-3B-TOFU-Steered-a-{alpha}"]
        assert build.depends_on == (
            "train-SmolLM3-3B-TOFU-Retain95-Arm",
            "train-SmolLM3-3B-TOFU-Forget05-Arm",
        )
        evaluation = plan.stage_index[f"eval-tofu-SmolLM3-3B-TOFU-Steered-a-{alpha}"]
        assert evaluation.depends_on == (build.id, oracle)
    assert plan.stage_index["eval-tofu-SmolLM3-3B-TOFU-Retain95-Oracle"].depends_on == (
        oracle,
    )


def test_tofu_training_and_steering_configs_are_matched() -> None:
    training = EXAMPLE / "configs" / "training" / "tofu"
    retain = yaml.safe_load((training / "arm-retain95.yml").read_text())
    forget = yaml.safe_load((training / "arm-forget05.yml").read_text())
    ignored = {"datasets", "dataset_prepared_path", "output_dir"}

    assert {key: value for key, value in retain.items() if key not in ignored} == {
        key: value for key, value in forget.items() if key not in ignored
    }
    assert retain["datasets"][0]["name"] == "retain95"
    assert forget["datasets"][0]["name"] == "forget05"
    assert retain["max_steps"] == forget["max_steps"] == 100
    assert retain["micro_batch_size"] * retain["gradient_accumulation_steps"] == 8
    for alpha in ("0.5", "1", "2"):
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs"
                / "steering"
                / "tofu"
                / f"retain-minus-forget-alpha-{alpha}.yml"
            ).read_text()
        )
        pair = config["adapter_pairs"][0]
        assert "Retain95-Arm" in pair["pos_adapter_name_or_path"]
        assert "Forget05-Arm" in pair["neg_adapter_name_or_path"]
        assert config["steered_adapters"][0]["alpha"] == float(alpha)


def test_tofu_metric_definitions_on_synthetic_inputs() -> None:
    metrics = _load_metrics()

    assert metrics.qa_probability(0) == 1
    assert metrics.rouge_l_recall("one two three", "one three") == pytest.approx(2 / 3)
    assert metrics.truth_ratio(2, [1, 1]) == pytest.approx(2.718281828)
    assert metrics.forget_truth_ratio(2) == pytest.approx(0.5)
    assert metrics.normalized_truth_ratio(0.25) == 0.75
    assert metrics.forget_quality([1, 2, 3], [1, 2, 3]) == 1
    assert metrics.model_utility([0.25, 1]) == pytest.approx(0.4)
    assert metrics.extraction_strength([0, 8, 2, 3], [1, 8, 2, 3]) == 0.75
