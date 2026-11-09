from pathlib import Path

import pytest

from lilpipe.cli import main
from conftest import _write_dependency_project
from test_results import _config


@pytest.mark.parametrize("prefix", [[], ["run"]])
def test_cli_supports_positional_config_and_plural_selections(project, capsys, prefix):
    result = main(
        [
            *prefix,
            "configs/experiments/pipeline.yml",
            "--models",
            "input",
            "--evaluations",
            "score",
            "--dry-run",
        ]
    )
    assert result == 0
    assert "eval-score-input" in capsys.readouterr().out


def test_cli_dry_run_reports_reused_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = main(
        [
            "configs/experiments/pipeline.yml",
            "--existing-models",
            "input",
            "--existing-models",
            "middle",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert output.startswith("# Reusing existing models: input, middle\n")
    assert "--job-name=produce-input" not in output
    assert "--job-name=produce-middle" not in output
    assert "--job-name=produce-output" in output


def test_cli_reports_pipeline_errors_without_traceback(project, capsys):
    result = main(
        ["configs/experiments/pipeline.yml", "--models", "missing", "--dry-run"]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "lilpipe: error: Unknown models" in captured.err
    assert "Traceback" not in captured.err


def test_cli_results_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "results.yml").write_text(_config())
    monkeypatch.chdir(tmp_path)
    assert main(["results", "results.yml", "--format", "json"]) == 0
    assert '"base"' in capsys.readouterr().out
    assert main(["results", "missing.yml"]) == 2
    assert "lilpipe: error:" in capsys.readouterr().err
