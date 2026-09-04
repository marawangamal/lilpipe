from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import lilpipe
from lilpipe import PipelineError, Plan, Stage
from lilpipe.cli import main


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "model-evaluation"
CONFIG = Path("configs/experiments/pipeline.yml")


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> lilpipe.Pipeline:
    monkeypatch.chdir(EXAMPLE)
    return lilpipe.load(CONFIG)


def test_full_example_builds_transitive_steered_model_plan(
    example: lilpipe.Pipeline,
) -> None:
    plan = example.plan()

    assert plan.id == "qwen3-8b-honesty-steering-comparison"
    assert len(plan.stages) == 7
    steering = plan.stage_index["build-qwen3-8b-honesty-steered"]
    assert steering.depends_on == (
        "train-qwen3-8b-honest-sft",
        "train-qwen3-8b-dishonest-sft",
    )
    assert plan.stage_index[
        "eval-truthfulqa-qwen3-8b-honesty-steered"
    ].depends_on == ("build-qwen3-8b-honesty-steered",)
    assert "eval-truthfulqa-qwen3-8b-honest-sft" not in plan.stage_index
    assert "eval-truthfulqa-qwen3-8b-dishonest-sft" not in plan.stage_index


def test_paths_are_resolved_from_cwd_not_pipeline_directory(
    example: lilpipe.Pipeline,
) -> None:
    assert example.root == EXAMPLE.resolve()
    rendered = example.select(
        models=["qwen3-8b-base"], evaluations=["truthfulqa"]
    ).plan().render()
    assert str(EXAMPLE / "scripts/eval_truthfulqa.sbatch") in rendered


def test_select_replaces_values_independently_without_mutation(
    example: lilpipe.Pipeline,
) -> None:
    models = example.select(models=["qwen3-8b-honesty-steered"])
    evaluations = example.select(evaluations=["truthfulqa"])

    assert models.selected_models == ("qwen3-8b-honesty-steered",)
    assert models.selected_evaluations == example.selected_evaluations
    assert evaluations.selected_models == example.selected_models
    assert evaluations.selected_evaluations == ("truthfulqa",)
    assert example.selected_models == (
        "qwen3-8b-base",
        "qwen3-8b-honesty-steered",
    )


def test_select_can_use_registered_evaluation_not_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, configured_evaluations=[])
    monkeypatch.chdir(tmp_path)

    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    plan = pipeline.select(evaluations=["score"]).plan()

    assert "eval-score-base" in plan.stage_index


def test_evaluations_keep_independent_contracts(example: lilpipe.Pipeline) -> None:
    plan = example.plan()
    truthfulqa = plan.stage_index["eval-truthfulqa-qwen3-8b-base"]
    humaneval = plan.stage_index["eval-humaneval-qwen3-8b-base"]

    assert truthfulqa.script == "scripts/eval_truthfulqa.sbatch"
    assert humaneval.script == "scripts/eval_humaneval.sbatch"
    assert truthfulqa.args == (
        "Qwen/Qwen3-8B",
        "none",
        "artifacts/evals/qwen3-8b-base/truthfulqa",
    )
    assert humaneval.args[-2:] == ("--temperature=0.2", "--samples=10")
    assert truthfulqa.sbatch_args != humaneval.sbatch_args


def test_skip_accepts_model_slugs_and_explicit_stage_ids(
    example: lilpipe.Pipeline,
) -> None:
    plan = example.plan(
        skip=[
            "qwen3-8b-honest-sft",
            "eval-truthfulqa-qwen3-8b-base",
        ]
    )

    assert plan.skipped == {
        "train-qwen3-8b-honest-sft",
        "eval-truthfulqa-qwen3-8b-base",
    }
    rendered = plan.render()
    assert "--job-name=train-qwen3-8b-honest-sft" not in rendered
    assert "--job-name=eval-truthfulqa-qwen3-8b-base" not in rendered
    assert "dry-run-train-qwen3-8b-honest-sft" not in rendered


def test_default_producer_id_and_model_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(
        tmp_path,
        models="""models:
  input:
    base_model: Qwen/Base
    producer:
      script: input.sbatch
  output:
    base_model: Qwen/Base
    producer:
      script: output.sbatch
      depends_on: [input]
""",
        configured_models=["output"],
    )
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").select(
        models=["output"]
    ).plan()

    assert plan.stage_index["produce-output"].depends_on == ("produce-input",)


def test_dependency_model_must_exist_and_have_a_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_project(
        tmp_path,
        models="""models:
  output:
    base_model: Qwen/Base
    producer:
      script: output.sbatch
      depends_on: [missing]
""",
        configured_models=["output"],
    )
    with pytest.raises(PipelineError, match="depends on unknown model"):
        lilpipe.load("configs/experiments/pipeline.yml").select(
            models=["output"]
        ).plan()

    _write_project(
        tmp_path,
        models="""models:
  input:
    base_model: Qwen/Base
  output:
    base_model: Qwen/Base
    producer:
      script: output.sbatch
      depends_on: [input]
""",
        configured_models=["output"],
    )
    with pytest.raises(PipelineError, match="has no producer"):
        lilpipe.load("configs/experiments/pipeline.yml").select(
            models=["output"]
        ).plan()


def test_plan_is_immutable_and_validates_programmatic_stages(tmp_path: Path) -> None:
    plan = Plan("valid", tmp_path, (Stage("job", "job.sbatch"),))
    with pytest.raises(FrozenInstanceError):
        plan.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.stage_index["other"] = Stage("other", "other.sbatch")  # type: ignore[index]
    with pytest.raises(PipelineError, match="Duplicate stage id"):
        Plan(
            "duplicate",
            tmp_path,
            (Stage("job", "a.sbatch"), Stage("job", "b.sbatch")),
        )
    with pytest.raises(PipelineError, match="unknown dependencies"):
        Plan(
            "unknown",
            tmp_path,
            (Stage("job", "job.sbatch", depends_on=("missing",)),),
        )


def test_plan_rejects_cycles(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="cycle"):
        Plan(
            "cycle",
            tmp_path,
            (
                Stage("a", "a.sbatch", depends_on=("b",)),
                Stage("b", "b.sbatch", depends_on=("a",)),
            ),
        )


def test_schema_version_and_unknown_keys_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, version=2)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PipelineError, match="version: 1"):
        lilpipe.load("configs/experiments/pipeline.yml")

    _write_project(tmp_path, extra_pipeline="unexpected: true\n")
    with pytest.raises(PipelineError, match="unknown keys.*unexpected"):
        lilpipe.load("configs/experiments/pipeline.yml")

    _write_project(
        tmp_path,
        models="""models:
  base:
    base_model: Qwen/Base
    producer:
      script: train.sbatch
      depend_on: [other]
""",
    )
    with pytest.raises(PipelineError, match="unknown keys.*depend_on"):
        lilpipe.load("configs/experiments/pipeline.yml")


def test_missing_template_field_and_duplicate_selection_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(
        tmp_path,
        evaluations="""evaluations:
  score:
    script: score.sbatch
    args: ['{model.adapter}']
""",
    )
    monkeypatch.chdir(tmp_path)
    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    with pytest.raises(PipelineError, match="has no field 'adapter'"):
        pipeline.plan()
    with pytest.raises(PipelineError, match="contains duplicates"):
        pipeline.select(models=["base", "base"])


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


def test_cli_supports_positional_config_and_plural_selections(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(EXAMPLE)

    result = main(
        [
            str(CONFIG),
            "--models",
            "qwen3-8b-base",
            "qwen3-8b-honesty-steered",
            "--evaluations",
            "truthfulqa",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "eval-truthfulqa-qwen3-8b-base" in output
    assert "eval-truthfulqa-qwen3-8b-honesty-steered" in output
    assert "humaneval" not in output


def test_cli_reports_pipeline_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(EXAMPLE)

    result = main([str(CONFIG), "--models", "missing", "--dry-run"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "lilpipe: error: Unknown models" in captured.err
    assert "Traceback" not in captured.err


def _write_project(
    root: Path,
    *,
    version: int = 1,
    configured_evaluations: list[str] | None = None,
    configured_models: list[str] | None = None,
    models: str = """models:
  base:
    base_model: Qwen/Base
""",
    evaluations: str = """evaluations:
  score:
    script: score.sbatch
    args: ['{model.base_model}']
""",
    extra_pipeline: str = "",
) -> None:
    registry_dir = root / "configs" / "registries"
    experiment_dir = root / "configs" / "experiments"
    registry_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "models.yml").write_text(models)
    (registry_dir / "evals.yml").write_text(evaluations)
    selected = ["score"] if configured_evaluations is None else configured_evaluations
    selected_models = ["base"] if configured_models is None else configured_models
    model_lines = "".join(f"  - {item}\n" for item in selected_models)
    evaluation_lines = "".join(f"  - {item}\n" for item in selected)
    (experiment_dir / "pipeline.yml").write_text(
        f"""version: {version}
id: test
registries:
  models: configs/registries/models.yml
  evaluations: configs/registries/evals.yml
models:
{model_lines}evaluations:
{evaluation_lines}{extra_pipeline}"""
    )
