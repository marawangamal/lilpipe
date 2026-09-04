from pathlib import Path

import pytest

from lilpipe import Pipeline, PipelineError, Stage, load_pipeline, submit_pipeline
from lilpipe.cli import main


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "model-evaluation"


def test_full_example_builds_transitive_steered_model_dag() -> None:
    pipeline = load_pipeline(EXAMPLE / "pipeline.yml", root=EXAMPLE)

    assert pipeline.id == "qwen3-8b-honesty-steering-comparison"
    assert len(pipeline.stages) == 7
    steering = pipeline.stages["build-qwen3-8b-honesty-steered"]
    assert steering.depends_on == (
        "train-qwen3-8b-honest-sft",
        "train-qwen3-8b-dishonest-sft",
    )
    assert pipeline.stages[
        "eval-truthfulqa-qwen3-8b-honesty-steered"
    ].depends_on == ("build-qwen3-8b-honesty-steered",)
    assert "eval-truthfulqa-qwen3-8b-honest-sft" not in pipeline.stages
    assert "eval-truthfulqa-qwen3-8b-dishonest-sft" not in pipeline.stages


def test_evaluations_have_independent_contracts() -> None:
    pipeline = load_pipeline(EXAMPLE / "pipeline.yml", root=EXAMPLE)
    truthfulqa = pipeline.stages["eval-truthfulqa-qwen3-8b-base"]
    humaneval = pipeline.stages["eval-humaneval-qwen3-8b-base"]

    assert truthfulqa.script == "scripts/eval_truthfulqa.sbatch"
    assert humaneval.script == "scripts/eval_humaneval.sbatch"
    assert truthfulqa.args == (
        "Qwen/Qwen3-8B",
        "none",
        "artifacts/evals/qwen3-8b-base/truthfulqa",
    )
    assert humaneval.args[-2:] == ("--temperature=0.2", "--samples=10")
    assert truthfulqa.sbatch_args != humaneval.sbatch_args


def test_model_override_retains_transitive_producers() -> None:
    pipeline = load_pipeline(
        EXAMPLE / "pipeline.yml",
        root=EXAMPLE,
        selected_models=["qwen3-8b-honesty-steered"],
        selected_evals=["truthfulqa"],
    )

    assert set(pipeline.stages) == {
        "train-qwen3-8b-honest-sft",
        "train-qwen3-8b-dishonest-sft",
        "build-qwen3-8b-honesty-steered",
        "eval-truthfulqa-qwen3-8b-honesty-steered",
    }


def test_submit_fans_in_and_applies_extra_args_after_stage_args() -> None:
    stages = {
        "left": Stage("left", "left.sbatch"),
        "right": Stage("right", "right.sbatch"),
        "join": Stage(
            "join",
            "join.sbatch",
            sbatch_args=("--time=01:00:00",),
            depends_on=("left", "right"),
        ),
    }
    commands: list[list[str]] = []

    def run(command: list[str]) -> str:
        commands.append(command)
        return str(len(commands))

    ids = submit_pipeline(
        Pipeline("fan-in", stages),
        root=ROOT,
        extra_sbatch_args=["--time=02:00:00", "--exclusive"],
        run=run,
    )

    assert ids == {"left": "1", "right": "2", "join": "3"}
    join = commands[2]
    assert "--dependency=afterok:1:2" in join
    assert join.index("--time=01:00:00") < join.index("--time=02:00:00")
    assert join.index("--exclusive") < join.index(str(ROOT / "join.sbatch"))


def test_dry_run_does_not_invoke_runner() -> None:
    pipeline = Pipeline("dry", {"job": Stage("job", "job.sbatch")})

    ids = submit_pipeline(
        pipeline,
        root=ROOT,
        dry_run=True,
        run=lambda command: pytest.fail(f"runner called with {command}"),
    )

    assert ids == {"job": "dry-run-job"}


def test_skipped_dependency_is_treated_as_satisfied() -> None:
    pipeline = Pipeline(
        "skip",
        {
            "first": Stage("first", "first.sbatch"),
            "second": Stage("second", "second.sbatch", depends_on=("first",)),
        },
    )
    commands: list[list[str]] = []

    submit_pipeline(
        pipeline,
        root=ROOT,
        skip=["first"],
        run=lambda command: commands.append(command) or "2",
    )

    assert len(commands) == 1
    assert not any(argument.startswith("--dependency=") for argument in commands[0])


def test_unknown_template_field_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "models.yml").write_text("models:\n  base:\n    base_model: Qwen/Base\n")
    (tmp_path / "evals.yml").write_text(
        "evaluations:\n  score:\n    script: score.sbatch\n"
        "    args: ['{model.adapter}']\n"
    )
    (tmp_path / "pipeline.yml").write_text(
        "id: missing-field\nregistries:\n  models: models.yml\n"
        "  evaluations: evals.yml\nmodels: [base]\nevaluations: [score]\n"
    )

    with pytest.raises(PipelineError, match="has no field 'adapter'"):
        load_pipeline(tmp_path / "pipeline.yml", root=tmp_path)


def test_producer_cycle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "models.yml").write_text(
        "models:\n"
        "  a:\n    producer:\n      id: build-a\n      script: a.sbatch\n"
        "      depends_on: [build-b]\n"
        "  b:\n    producer:\n      id: build-b\n      script: b.sbatch\n"
        "      depends_on: [build-a]\n"
    )
    (tmp_path / "evals.yml").write_text("evaluations: {}\n")
    (tmp_path / "pipeline.yml").write_text(
        "id: cycle\nregistries:\n  models: models.yml\n"
        "  evaluations: evals.yml\nmodels: [a]\nevaluations: []\n"
    )

    with pytest.raises(PipelineError, match="cycle"):
        load_pipeline(tmp_path / "pipeline.yml", root=tmp_path)


def test_unknown_model_and_evaluation_are_rejected() -> None:
    with pytest.raises(PipelineError, match="Unknown models"):
        load_pipeline(
            EXAMPLE / "pipeline.yml",
            root=EXAMPLE,
            selected_models=["missing-model"],
        )
    with pytest.raises(PipelineError, match="Unknown evaluations"):
        load_pipeline(
            EXAMPLE / "pipeline.yml",
            root=EXAMPLE,
            selected_evals=["missing-eval"],
        )


def test_conflicting_producer_definitions_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "models.yml").write_text(
        "models:\n"
        "  a:\n    producer:\n      id: shared\n      script: a.sbatch\n"
        "  b:\n    producer:\n      id: shared\n      script: b.sbatch\n"
    )
    (tmp_path / "evals.yml").write_text("evaluations: {}\n")
    (tmp_path / "pipeline.yml").write_text(
        "id: conflict\nregistries:\n  models: models.yml\n"
        "  evaluations: evals.yml\nmodels: [a]\nevaluations: []\n"
    )

    with pytest.raises(PipelineError, match="conflicting definitions"):
        load_pipeline(tmp_path / "pipeline.yml", root=tmp_path)


def test_unknown_producer_dependency_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "models.yml").write_text(
        "models:\n  a:\n    producer:\n      id: build-a\n"
        "      script: a.sbatch\n      depends_on: [missing]\n"
    )
    (tmp_path / "evals.yml").write_text("evaluations: {}\n")
    (tmp_path / "pipeline.yml").write_text(
        "id: unknown-dependency\nregistries:\n  models: models.yml\n"
        "  evaluations: evals.yml\nmodels: [a]\nevaluations: []\n"
    )

    with pytest.raises(PipelineError, match="Unknown producer dependency"):
        load_pipeline(tmp_path / "pipeline.yml", root=tmp_path)


def test_cli_dry_run_and_filters(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--config",
            "pipeline.yml",
            "--root",
            str(EXAMPLE),
            "--models",
            "qwen3-8b-base",
            "--only-eval",
            "truthfulqa",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "eval-truthfulqa-qwen3-8b-base" in output
    assert "humaneval" not in output
    assert "train-" not in output
