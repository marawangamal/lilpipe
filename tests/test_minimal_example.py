import json
from pathlib import Path

import pytest
import yaml

import lilpipe
from lilpipe.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "minimal"
PIPELINE_CONFIG = Path("configs/experiments/pipeline.yml")
RESULTS_CONFIG = EXAMPLE / "configs" / "results" / "main.yml"


def test_minimal_example_loads_and_referenced_files_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load(PIPELINE_CONFIG)

    manifest = yaml.safe_load(PIPELINE_CONFIG.read_text())
    for registry in manifest["registries"].values():
        assert Path(registry).is_file()
    for stage in pipeline.plan().stages:
        assert Path(stage.script).is_file()

    assert pipeline.id == "minimal-toy-lineage"
    assert pipeline.selected_models == ("toy-tuned",)
    assert pipeline.selected_evaluations == ("toy-score",)


def test_minimal_example_plans_transitive_lineage_and_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(EXAMPLE)

    plan = lilpipe.load(PIPELINE_CONFIG).plan()

    assert tuple(stage.id for stage in plan.stages) == (
        "produce-toy-base",
        "produce-toy-tuned",
        "eval-toy-score-toy-tuned",
    )
    assert plan.stage_index["produce-toy-tuned"].depends_on == ("produce-toy-base",)
    evaluation = plan.stage_index["eval-toy-score-toy-tuned"]
    assert evaluation.depends_on == ("produce-toy-tuned",)
    assert evaluation.script == "scripts/evaluate.sbatch"
    assert evaluation.args == (
        "artifacts/models/toy-tuned",
        "artifacts/evals/toy-tuned/toy-score",
        "Toy tuned",
    )


def test_minimal_example_cli_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(EXAMPLE)

    assert main([str(PIPELINE_CONFIG), "--dry-run"]) == 0

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert len(lines) == 3
    assert str(EXAMPLE / "scripts" / "produce.sbatch") in lines[0]
    assert "--job-name=produce-toy-base" in lines[0]
    assert "--job-name=produce-toy-tuned" in lines[1]
    assert "--dependency=afterok:dry-run-produce-toy-base" in lines[1]
    assert str(EXAMPLE / "scripts" / "evaluate.sbatch") in lines[2]
    assert "--dependency=afterok:dry-run-produce-toy-tuned" in lines[2]
    assert "artifacts/models/toy-tuned" in lines[2]


def test_minimal_results_config_collects_lm_eval_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifacts/evals/toy-tuned/toy-score/results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"results": {"toy_task": {"acc,none": 0.75}}}))
    config = tmp_path / "configs/results/main.yml"
    config.parent.mkdir(parents=True)
    config.write_text(RESULTS_CONFIG.read_text())
    monkeypatch.chdir(tmp_path)

    rendered = lilpipe.load_results("configs/results/main.yml").collect().render()

    assert "Toy base" in rendered
    assert "—" in rendered
    assert "Toy tuned" in rendered
    assert "**75.0%**" in rendered
