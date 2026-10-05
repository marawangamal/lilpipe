"""Regression coverage for independently weighted WS arms."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "examples" / "tamper-resistance"


def load_steering_module():
    path = ROOT / "scripts/weight_steering.py"
    spec = importlib.util.spec_from_file_location("tamper_weight_steering", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
