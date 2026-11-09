from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import lilpipe
from lilpipe import PipelineError, Plan, Stage
from conftest import _write_project, _write_dependency_project


def test_paths_are_resolved_from_cwd_not_pipeline_directory(project):
    assert project.root == Path.cwd()
    assert str(Path.cwd() / "output.sbatch") in project.plan().render()


def test_select_replaces_values_independently_without_mutation(project):
    models = project.select(models=["input"])
    evaluations = project.select(evaluations=[])
    assert models.selected_models == ("input",)
    assert models.selected_evaluations == project.selected_evaluations
    assert evaluations.selected_models == project.selected_models
    assert evaluations.selected_evaluations == ()
    assert project.selected_models == ("output",)
    assert project.selected_evaluations == ("score",)


def test_select_can_use_registered_evaluation_not_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, configured_evaluations=[])
    monkeypatch.chdir(tmp_path)

    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    plan = pipeline.select(evaluations=["score"]).plan()

    assert "eval-score-base" in plan.stage_index


def test_skip_accepts_model_slugs_and_explicit_stage_ids(project):
    plan = project.plan(skip=["input", "eval-score-output"])
    assert plan.skipped == {"produce-input", "eval-score-output"}
    assert "--job-name=produce-input" not in plan.render()
    assert "--job-name=eval-score-output" not in plan.render()


def test_existing_model_skips_producer_but_keeps_transitive_downstream_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan(
        existing_models=["input"]
    )

    assert plan.skipped == {"produce-input"}
    assert tuple(stage.id for stage in plan.stages) == (
        "produce-input",
        "produce-middle",
        "produce-output",
        "eval-score-output",
    )
    commands = plan.render().splitlines()
    middle = next(line for line in commands if "--job-name=produce-middle" in line)
    output = next(line for line in commands if "--job-name=produce-output" in line)
    assert "--dependency=" not in middle
    assert "--dependency=afterok:dry-run-produce-middle" in output


def test_existing_selected_model_is_still_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan(
        existing_models=["output"]
    )

    rendered = plan.render()
    assert "--job-name=produce-output" not in rendered
    evaluation = next(
        line for line in rendered.splitlines() if "--job-name=eval-score-output" in line
    )
    assert "--dependency=" not in evaluation


def test_existing_models_and_skip_are_combined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan(
        existing_models=["input", "middle"], skip=["produce-output"]
    )

    assert plan.skipped == {"produce-input", "produce-middle", "produce-output"}
    assert "--job-name=eval-score-output" in plan.render()


@pytest.mark.parametrize(
    ("existing_models", "message"),
    [
        (["missing"], "Unknown existing model: 'missing'"),
        (["external"], "Existing model 'external' has no producer to skip"),
        (["input", "input"], "existing_models contains duplicates: \\['input'\\]"),
        (["irrelevant"], "not in the active selection's dependency closure"),
    ],
)
def test_existing_models_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_models: list[str],
    message: str,
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PipelineError, match=message):
        lilpipe.load("configs/experiments/pipeline.yml").plan(
            existing_models=existing_models
        )


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

    plan = (
        lilpipe.load("configs/experiments/pipeline.yml")
        .select(models=["output"])
        .plan()
    )

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


def test_evaluation_dependencies_are_deduplicated_and_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(
        tmp_path,
        models="""models:
  oracle:
    base_model: Qwen/Oracle
    producer:
      script: oracle.sbatch
  candidate:
    base_model: Qwen/Candidate
    producer:
      script: candidate.sbatch
""",
        evaluations="""evaluations:
  score:
    script: score.sbatch
    args: ['{model.base_model}']
    depends_on: [oracle, oracle]
""",
        configured_models=["candidate"],
    )
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan()

    assert plan.stage_index["eval-score-candidate"].depends_on == (
        "produce-candidate",
        "produce-oracle",
    )
    assert tuple(stage.id for stage in plan.stages) == (
        "produce-oracle",
        "produce-candidate",
        "eval-score-candidate",
    )


@pytest.mark.parametrize(
    ("dependency_model", "message"),
    [("missing", "unknown model"), ("base", "has no producer")],
)
def test_evaluation_dependency_must_exist_and_have_a_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency_model: str,
    message: str,
) -> None:
    _write_project(
        tmp_path,
        evaluations=f"""evaluations:
  score:
    script: score.sbatch
    depends_on: [{dependency_model}]
""",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PipelineError, match=message):
        lilpipe.load("configs/experiments/pipeline.yml").plan()

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
