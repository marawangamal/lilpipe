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
        'export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-open-unlearning-$SLURM_JOB_ID"'
        in script
    )
    assert "uv pip install --offline" in script
    assert "snapshots/$revision" in script
    if script_name == "unlearn_wmdp.sbatch":
        assert (
            'bio_files=("$bio_cache/snapshots/$bio_revision/data/"*.parquet)' in script
        )
        assert "scripts/lilpipe/rmu_bio.py" in script
        assert "wikitext-2-raw-v1-test.parquet" in script


def test_wmdp_bio_rmu_example_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(EXAMPLE_ROOT)
    plan = load("configs/lilpipe/experiments/wmdp-bio-rmu.yml").plan()

    assert [stage.id for stage in plan.stages] == [
        "unlearn-zephyr-7b-beta-rmu-bio",
        "eval-wmdp-bio-and-mmlu-no-bio-zephyr-7b-beta",
        "eval-wmdp-bio-and-mmlu-no-bio-zephyr-7b-beta-rmu-bio",
    ]
    assert plan.stages[-1].args[-2:] == ("bio", "mmlu_no_bio")
    assert "--gres=gpu:l40s:2" in plan.stages[0].sbatch_args
    assert "--mem=64G" in plan.stages[0].sbatch_args


def test_wmdp_bio_uses_paper_rmu_configuration() -> None:
    script = (EXAMPLE_ROOT / "scripts/lilpipe/rmu_bio.py").read_text()

    assert "default=150" in script
    assert "default=4" in script
    assert "default=6.5" in script
    assert "default=1200.0" in script
    assert "default=5e-5" in script
    assert "mlp.down_proj.weight" in script
