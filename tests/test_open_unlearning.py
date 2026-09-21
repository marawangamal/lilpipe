from pathlib import Path

import pytest

from lilpipe import load

EXAMPLE_ROOT = Path(__file__).parents[1] / "examples" / "open-unlearning"


def test_wmdp_rmu_example_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(EXAMPLE_ROOT)
    pipeline = load("configs/lilpipe/experiments/wmdp-cyber-rmu.yml")

    plan = pipeline.plan()

    assert [stage.id for stage in plan.stages] == [
        "unlearn-zephyr-7b-beta-rmu-cyber",
        "eval-wmdp-cyber-and-mmlu-zephyr-7b-beta",
        "eval-wmdp-cyber-and-mmlu-zephyr-7b-beta-rmu-cyber",
    ]
    assert plan.stages[-1].depends_on == ("unlearn-zephyr-7b-beta-rmu-cyber",)
