from pathlib import Path

import pytest

import lilpipe


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
    (experiment_dir / "pipeline.yml").write_text(f"""version: {version}
id: test
registries:
  models: configs/registries/models.yml
  evaluations: configs/registries/evals.yml
models:
{model_lines}evaluations:
{evaluation_lines}{extra_pipeline}""")


def _write_dependency_project(root: Path) -> None:
    _write_project(
        root,
        models="""models:
  external:
    base_model: Remote/Model
  input:
    base_model: Local/Input
    producer:
      script: input.sbatch
  middle:
    base_model: Local/Middle
    producer:
      script: middle.sbatch
      depends_on: [input]
  output:
    base_model: Local/Output
    producer:
      script: output.sbatch
      depends_on: [middle]
  irrelevant:
    base_model: Local/Irrelevant
    producer:
      script: irrelevant.sbatch
""",
        configured_models=["output"],
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    return lilpipe.load("configs/experiments/pipeline.yml")
