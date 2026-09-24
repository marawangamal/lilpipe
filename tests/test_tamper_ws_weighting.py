"""Regression coverage for independently weighted WS arms."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1] / "examples" / "tamper-resistance"


def load_steering_module():
    path = ROOT / "scripts/weight_steering.py"
    spec = importlib.util.spec_from_file_location("tamper_weight_steering", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_separate_weights_match_requested_grad_diff_ratio() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/unlearn/di-6.9b/wmdp-bio-unlearn-ws-r1-f01.yml").read_text()
    )
    assert config["base_model_name_or_path"] == ("EleutherAI/deep-ignorance-unfiltered")
    assert config["adapter_pairs"] == [
        {
            "pos_adapter_name_or_path": (
                "artifacts/models/di-6.9b-wmdp-bio-unlearn-ws-ft-retain"
            ),
            "neg_adapter_name_or_path": (
                "artifacts/models/di-6.9b-wmdp-bio-unlearn-ws-ft-forget"
            ),
        }
    ]
    steered = config["steered_adapters"][0]
    assert steered["output_path"].endswith("di-6.9b-wmdp-bio-unlearn-ws-r1-f01")
    assert load_steering_module().weighted_adapter_spec(steered) == (
        "steered_r1_f0_1",
        [1.0, -0.1],
    )


def test_existing_alpha_configs_keep_equal_opposite_weights() -> None:
    steering = load_steering_module()
    assert steering.weighted_adapter_spec({"alpha": 2.0}) == (
        "steered_2",
        [2.0, -2.0],
    )
    with pytest.raises(ValueError, match="either alpha or separate"):
        steering.weighted_adapter_spec(
            {"alpha": 1.0, "retain_weight": 1.0, "forget_weight": 0.1}
        )
