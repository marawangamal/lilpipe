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


@pytest.mark.parametrize("script_name", ["unlearn_wmdp.sbatch", "eval_wmdp.sbatch"])
def test_wmdp_jobs_load_torch_cuda_toolkit(script_name: str) -> None:
    script = (EXAMPLE_ROOT / "scripts/lilpipe" / script_name).read_text()

    assert "module load cuda/12.1.1" in script
    assert 'export HF_HUB_CACHE="$HF_HOME/hub"' in script
    assert (
        'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-open-unlearning"' in script
    )
    assert "uv pip install --offline" in script
    assert "snapshots/$revision" in script
