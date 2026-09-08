from pathlib import Path

import pytest
import yaml

import lilpipe


ROOT = Path(__file__).resolve().parents[1] / "examples" / "weight-steering"

CONTROL_MODELS = (
    "SmolLM3-3B-FT-Sycophantic",
    "SmolLM3-3B-FT-Non-Sycophantic",
    "SmolLM3-3B-W-Steer-Non-Sycophancy-a--4",
    "SmolLM3-3B-W-Steer-Non-Sycophancy-a--1",
    "SmolLM3-3B-W-Steer-Non-Sycophancy-a-1",
    "SmolLM3-3B-W-Steer-Non-Sycophancy-a-4",
)


def test_base_sycophancy_training_configs_are_matched() -> None:
    paths = (
        ROOT / "configs/training/base-sycophantic.yml",
        ROOT / "configs/training/base-non-sycophantic.yml",
    )
    sycophantic, non_sycophantic = [yaml.safe_load(path.read_text()) for path in paths]
    assert sycophantic["datasets"][0]["path"] == "cfierro/pv-prompts-sycophantic"
    assert non_sycophantic["datasets"][0]["path"] == (
        "cfierro/pv-prompts-non-sycophantic"
    )
    for config in (sycophantic, non_sycophantic):
        assert config["base_model"] == "HuggingFaceTB/SmolLM3-3B"
        assert config["output_dir"].startswith("artifacts/models/SmolLM3-3B-FT-")
        config["datasets"][0]["path"] = "PAIR"
        config["dataset_prepared_path"] = "PAIR"
        config["output_dir"] = "PAIR"
    assert sycophantic == non_sycophantic


@pytest.mark.parametrize(
    ("token", "alpha", "suffix"),
    (("neg-4", -4.0, "a--4"), ("neg-1", -1.0, "a--1"),
     ("1", 1.0, "a-1"), ("4", 4.0, "a-4")),
)
def test_base_sycophancy_steering_configs(
    token: str, alpha: float, suffix: str
) -> None:
    path = ROOT / f"configs/steering/base-non-sycophancy-alpha-{token}.yml"
    config = yaml.safe_load(path.read_text())
    pair = config["adapter_pairs"][0]
    assert config["base_model_name_or_path"] == "HuggingFaceTB/SmolLM3-3B"
    assert pair == {
        "pos_adapter_name_or_path": "artifacts/models/SmolLM3-3B-FT-Non-Sycophantic",
        "neg_adapter_name_or_path": "artifacts/models/SmolLM3-3B-FT-Sycophantic",
    }
    assert config["steered_adapters"] == [{
        "alpha": alpha,
        "output_path": f"artifacts/models/SmolLM3-3B-W-Steer-Non-Sycophancy-{suffix}",
    }]


def test_focused_base_sycophancy_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    assert set(CONTROL_MODELS) <= set(pipeline.models)
    plan = pipeline.select(
        models=("SmolLM3-3B", *CONTROL_MODELS), evaluations=("mbpp",)
    ).plan()

    evaluations = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    assert len(evaluations) == 7
    assert "train-SmolLM3-3B-FT-Sycophantic" in plan.stage_index
    assert "train-SmolLM3-3B-FT-Non-Sycophantic" in plan.stage_index
    assert "train-SmolLM3-3B-HMO" not in plan.stage_index
    for suffix in ("a--4", "a--1", "a-1", "a-4"):
        stage = plan.stage_index[
            f"build-SmolLM3-3B-W-Steer-Non-Sycophancy-{suffix}"
        ]
        assert set(stage.depends_on) == {
            "train-SmolLM3-3B-FT-Sycophantic",
            "train-SmolLM3-3B-FT-Non-Sycophantic",
        }
        assert all("HMO" not in dependency for dependency in stage.depends_on)


def test_hmo_derived_direction_is_applied_to_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "SmolLM3-3B-W-Steer-HMO-Derived-Non-Sycophancy-a--4"
    config = yaml.safe_load(
        (
            ROOT
            / "configs/steering/base-hmo-derived-non-sycophancy-alpha-neg-4.yml"
        ).read_text()
    )
    assert config["base_model_name_or_path"] == "HuggingFaceTB/SmolLM3-3B"
    assert config["adapter_pairs"] == [{
        "pos_adapter_name_or_path": (
            "artifacts/models/SmolLM3-3B-HMO-FT-Non-Sycophantic"
        ),
        "neg_adapter_name_or_path": (
            "artifacts/models/SmolLM3-3B-HMO-FT-Sycophantic"
        ),
    }]
    assert config["steered_adapters"] == [{
        "alpha": -4.0,
        "output_path": f"artifacts/models/{model}",
    }]

    monkeypatch.chdir(ROOT)
    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    plan = pipeline.select(models=(model,), evaluations=("mbpp",)).plan(
        skip=(
            "SmolLM3-3B-HMO-FT-Sycophantic",
            "SmolLM3-3B-HMO-FT-Non-Sycophantic",
        )
    )
    build = plan.stage_index[f"build-{model}"]
    assert set(build.depends_on) == {
        "train-SmolLM3-3B-HMO-FT-Sycophantic",
        "train-SmolLM3-3B-HMO-FT-Non-Sycophantic",
    }
    assert set(build.depends_on) <= plan.skipped
    assert plan.stage_index[f"eval-mbpp-{model}"].depends_on == (f"build-{model}",)
