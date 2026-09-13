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
    plan = lilpipe.load("configs/experiments/smollm3/main-tofu.yml").plan()

    assert len(plan.stages) == 14
    oracle = "train-SmolLM3-3B-TOFU95"
    for alpha in ("0.5", "1", "2"):
        build = plan.stage_index[f"build-SmolLM3-3B-TOFU100-W-Steer-Forget-a-{alpha}"]
        assert build.depends_on == (
            "train-SmolLM3-3B-TOFU100-FT-Retain95",
            "train-SmolLM3-3B-TOFU100-FT-Forget05",
        )
        evaluation = plan.stage_index[
            f"eval-tofu-SmolLM3-3B-TOFU100-W-Steer-Forget-a-{alpha}"
        ]
        assert evaluation.depends_on == (build.id, oracle)
    assert plan.stage_index["eval-tofu-SmolLM3-3B-TOFU95"].depends_on == (oracle,)
    for stage in plan.stages:
        exclusion = next(
            argument
            for argument in stage.sbatch_args
            if argument.startswith("--exclude=")
        )
        assert {
            "cn-b001",
            "cn-b002",
            "cn-b003",
            "cn-b004",
            "cn-b005",
            "cn-e002",
            "cn-e003",
        } <= set(exclusion.removeprefix("--exclude=").split(","))


def test_tofu_training_and_steering_configs_are_matched() -> None:
    training = EXAMPLE / "configs" / "training" / "smollm3"
    retain = yaml.safe_load((training / "tofu-retain95.yml").read_text())
    forget = yaml.safe_load((training / "tofu-forget05.yml").read_text())
    ignored = {"datasets", "dataset_prepared_path", "output_dir", "wandb_name"}

    assert {key: value for key, value in retain.items() if key not in ignored} == {
        key: value for key, value in forget.items() if key not in ignored
    }
    assert retain["datasets"][0]["name"] == "retain95"
    assert forget["datasets"][0]["name"] == "forget05"
    assert retain["datasets"][0]["type"] == "scripts.data.tofu"
    assert retain["max_steps"] == forget["max_steps"] == 100
    assert retain["micro_batch_size"] * retain["gradient_accumulation_steps"] == 8
    for alpha in ("0.5", "1", "2"):
        config = yaml.safe_load(
            (
                EXAMPLE
                / "configs"
                / "steering"
                / "smollm3"
                / f"tofu-retain-minus-forget-alpha-{alpha}.yml"
            ).read_text()
        )
        pair = config["adapter_pairs"][0]
        assert "TOFU100-FT-Retain95" in pair["pos_adapter_name_or_path"]
        assert "TOFU100-FT-Forget05" in pair["neg_adapter_name_or_path"]
        assert config["steered_adapters"][0]["alpha"] == float(alpha)


def test_tofu_training_configs_use_canonical_wandb_runs() -> None:
    training = EXAMPLE / "configs" / "training" / "smollm3"
    expected = {
        "tofu-target.yml": "SmolLM3-3B-TOFU100",
        "tofu-oracle95.yml": "SmolLM3-3B-TOFU95",
        "tofu-retain95.yml": "SmolLM3-3B-TOFU100-FT-Retain95",
        "tofu-forget05.yml": "SmolLM3-3B-TOFU100-FT-Forget05",
    }

    for filename, run_name in expected.items():
        config = yaml.safe_load((training / filename).read_text())
        assert config["wandb_project"] == "lp-weight-steering"
        assert config["wandb_name"] == run_name


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
