from pathlib import Path

import pytest

import lilpipe
from lilpipe import Plan, Stage
from lilpipe.cli import main


def test_submit_fans_in_and_applies_extra_args_after_stage_args(
    tmp_path: Path,
) -> None:
    plan = Plan(
        "fan-in",
        tmp_path,
        (
            Stage("left", "left.sbatch"),
            Stage("right", "right.sbatch"),
            Stage(
                "join",
                "join.sbatch",
                sbatch_args=("--time=01:00:00",),
                depends_on=("left", "right"),
            ),
        ),
    )
    commands: list[list[str]] = []

    def run(command: list[str]) -> str:
        commands.append(command)
        return str(len(commands))

    ids = lilpipe.submit(
        plan,
        extra_sbatch_args=["--time=02:00:00", "--exclusive"],
        run=run,
    )

    assert ids == {"left": "1", "right": "2", "join": "3"}
    join = commands[2]
    assert "--dependency=afterok:1:2" in join
    assert join.index("--time=01:00:00") < join.index("--time=02:00:00")
    assert join.index("--exclusive") < join.index(str(tmp_path / "join.sbatch"))


@pytest.mark.parametrize("fail_second", [False, True])
def test_recorded_submission_preserves_sources_and_partial_job_ids(
    tmp_path, monkeypatch, fail_second
):
    import json
    import subprocess
    from types import SimpleNamespace
    import yaml
    import lilpipe.cli as cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.sbatch").write_text("#!/bin/bash\n")
    (tmp_path / "base.yml").write_text("learning_rate: 0.0001\n")
    models = {
        name: {
            "base_model_name_or_path": name,
            "producer": {
                "script": "train.sbatch",
                "args": ["base.yml", "--learning-rate", "0.0002"],
            },
        }
        for name in ["one", "two"]
    }
    (tmp_path / "models.yml").write_text(yaml.safe_dump({"models": models}))
    (tmp_path / "evals.yml").write_text("evaluations: {}\n")
    manifest = {
        "version": 1,
        "id": "record-test",
        "registries": {"models": "models.yml", "evaluations": "evals.yml"},
        "models": ["one", "two"],
        "evaluations": [],
    }
    (tmp_path / "experiment.yml").write_text(yaml.safe_dump(manifest))
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(
                stdout="version\n" if command[1] == "rev-parse" else ""
            )
        calls.append(command)
        if fail_second and len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=str(100 + len(calls)) + ";cluster\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert main(["experiment.yml", "--record-dir", "record", "--dry-run"]) == 0
    assert not (tmp_path / "record").exists()
    assert main(["experiment.yml", "--record-dir", "record"]) == (
        2 if fail_second else 0
    )
    record = json.loads((tmp_path / "record/jobs.json").read_text())
    assert record["jobs"][0]["job_id"] == "101"
    assert record["status"] == ("partial submission" if fail_second else "submitted")
    assert len(record["sources"]) == 5
    assert (tmp_path / "record/submitted/base.yml").read_bytes() == (
        tmp_path / "base.yml"
    ).read_bytes()
    assert main(["experiment.yml", "--record-dir", "record"]) == 2
    assert len(calls) == 2
