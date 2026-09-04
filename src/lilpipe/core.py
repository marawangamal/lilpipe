"""Compile model/evaluation registries and submit their Slurm DAGs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
import shlex
import subprocess

import yaml


class PipelineError(ValueError):
    """Raised when a pipeline cannot be compiled or submitted safely."""


@dataclass(frozen=True)
class Stage:
    id: str
    script: str
    args: tuple[str, ...] = ()
    sbatch_args: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pipeline:
    id: str
    stages: Mapping[str, Stage]


def _read_yaml(path: Path) -> object:
    try:
        with path.open() as handle:
            return yaml.safe_load(handle)
    except OSError as error:
        raise PipelineError(f"Could not read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise PipelineError(f"Could not parse {path}: {error}") from error


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_registry(root: Path, path: object, key: str) -> dict[str, object]:
    if not isinstance(path, str):
        raise PipelineError(f"Registry path for {key!r} must be a string")
    data = _read_yaml(_resolve(root, path))
    if not isinstance(data, Mapping) or not isinstance(data.get(key), Mapping):
        raise PipelineError(f"Registry {path!r} must contain a {key!r} mapping")
    return dict(data[key])


def _string_tuple(value: object, *, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineError(f"{location} must be a list")
    return tuple(str(item) for item in value)


def _producer_stage(model_id: str, model: Mapping[str, object]) -> Stage | None:
    raw = model.get("producer")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PipelineError(f"Model {model_id!r} producer must be a mapping")
    producer_id = raw.get("id")
    script = raw.get("script")
    if not isinstance(producer_id, str) or not producer_id:
        raise PipelineError(f"Model {model_id!r} producer must define a non-empty id")
    if not isinstance(script, str) or not script:
        raise PipelineError(f"Producer {producer_id!r} must define a non-empty script")
    return Stage(
        id=producer_id,
        script=script,
        args=_string_tuple(raw.get("args", ()), location=f"Producer {producer_id!r} args"),
        sbatch_args=_string_tuple(
            raw.get("sbatch_args", ()),
            location=f"Producer {producer_id!r} sbatch_args",
        ),
        depends_on=_string_tuple(
            raw.get("depends_on", ()),
            location=f"Producer {producer_id!r} depends_on",
        ),
    )


def _render_argument(template: object, *, model_id: str, model: Mapping[str, object]) -> str:
    value = str(template)
    fields = {"id": model_id, **model}
    pieces: list[str] = []
    try:
        parsed = Formatter().parse(value)
        for literal, field_name, format_spec, conversion in parsed:
            pieces.append(literal)
            if field_name is None:
                continue
            if format_spec or conversion:
                raise PipelineError(
                    f"Evaluation argument {value!r} cannot use conversions or format specs"
                )
            if not field_name.startswith("model."):
                raise PipelineError(
                    f"Evaluation argument {value!r} has unsupported placeholder "
                    f"{field_name!r}; use {{model.<field>}}"
                )
            key = field_name.removeprefix("model.")
            if not key or "." in key or key not in fields:
                raise PipelineError(
                    f"Model {model_id!r} has no field {key!r} required by {value!r}"
                )
            replacement = fields[key]
            if isinstance(replacement, (Mapping, Sequence)) and not isinstance(
                replacement, (str, bytes)
            ):
                raise PipelineError(
                    f"Model field {key!r} used by {value!r} must be a scalar"
                )
            pieces.append(str(replacement))
    except PipelineError:
        raise
    except ValueError as error:
        raise PipelineError(f"Invalid evaluation argument template {value!r}: {error}") from error
    return "".join(pieces)


def _validate_stages(stages: Mapping[str, Stage]) -> None:
    names = set(stages)
    for stage in stages.values():
        unknown = set(stage.depends_on) - names
        if unknown:
            raise PipelineError(
                f"Stage {stage.id!r} has unknown dependencies: {sorted(unknown)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            raise PipelineError(f"Dependency cycle includes stage {stage_id!r}")
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for dependency in stages[stage_id].depends_on:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in stages:
        visit(stage_id)


def load_pipeline(
    config_path: str | Path,
    *,
    root: str | Path,
    selected_models: Iterable[str] = (),
    selected_evals: Iterable[str] = (),
) -> Pipeline:
    """Compile a pipeline manifest and its registries into a validated DAG."""

    root = Path(root)
    data = _read_yaml(Path(config_path))
    if not isinstance(data, Mapping):
        raise PipelineError("Pipeline configuration must be a mapping")
    pipeline_id = data.get("id")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise PipelineError("Pipeline configuration must define a non-empty id")
    registries = data.get("registries")
    if not isinstance(registries, Mapping):
        raise PipelineError("Pipeline configuration must define registries")

    models = _load_registry(root, registries.get("models"), "models")
    evaluations = _load_registry(root, registries.get("evaluations"), "evaluations")

    manifest_models = _string_tuple(data.get("models", ()), location="Pipeline models")
    manifest_evals = _string_tuple(data.get("evaluations", ()), location="Pipeline evaluations")
    wanted_models = tuple(selected_models) or manifest_models
    requested_evals = tuple(selected_evals)

    unknown_models = set(wanted_models) - set(models)
    if unknown_models:
        raise PipelineError(f"Unknown models: {sorted(unknown_models)}")
    unknown_manifest_evals = set(manifest_evals) - set(evaluations)
    if unknown_manifest_evals:
        raise PipelineError(f"Unknown evaluations: {sorted(unknown_manifest_evals)}")
    unknown_requested_evals = set(requested_evals) - set(evaluations)
    if unknown_requested_evals:
        raise PipelineError(f"Unknown evaluations: {sorted(unknown_requested_evals)}")
    wanted_evals = (
        tuple(item for item in manifest_evals if item in set(requested_evals))
        if requested_evals
        else manifest_evals
    )

    model_mappings: dict[str, Mapping[str, object]] = {}
    producers: dict[str, Stage] = {}
    model_producers: dict[str, str] = {}
    for model_id, raw_model in models.items():
        if not isinstance(model_id, str) or not isinstance(raw_model, Mapping):
            raise PipelineError(f"Model {model_id!r} must be a mapping")
        model_mappings[model_id] = raw_model
        producer = _producer_stage(model_id, raw_model)
        if producer is None:
            continue
        existing = producers.get(producer.id)
        if existing is not None and existing != producer:
            raise PipelineError(f"Producer {producer.id!r} has conflicting definitions")
        producers[producer.id] = producer
        model_producers[model_id] = producer.id

    needed = {
        model_producers[model_id]
        for model_id in wanted_models
        if model_id in model_producers
    }
    pending = list(needed)
    while pending:
        producer_id = pending.pop()
        producer = producers.get(producer_id)
        if producer is None:
            raise PipelineError(f"Unknown producer dependency: {producer_id!r}")
        for dependency in producer.depends_on:
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)

    stages: dict[str, Stage] = {
        producer_id: producer
        for producer_id, producer in producers.items()
        if producer_id in needed
    }

    for model_id in wanted_models:
        model = model_mappings[model_id]
        for evaluation_id in wanted_evals:
            raw_evaluation = evaluations[evaluation_id]
            if not isinstance(raw_evaluation, Mapping):
                raise PipelineError(f"Evaluation {evaluation_id!r} must be a mapping")
            script = raw_evaluation.get("script")
            if not isinstance(script, str) or not script:
                raise PipelineError(
                    f"Evaluation {evaluation_id!r} must define a non-empty script"
                )
            raw_args = raw_evaluation.get("args", ())
            args = _string_tuple(raw_args, location=f"Evaluation {evaluation_id!r} args")
            stage_id = f"eval-{evaluation_id}-{model_id}"
            if stage_id in stages:
                raise PipelineError(f"Duplicate stage id: {stage_id!r}")
            dependency = model_producers.get(model_id)
            stages[stage_id] = Stage(
                id=stage_id,
                script=script,
                args=tuple(
                    _render_argument(argument, model_id=model_id, model=model)
                    for argument in args
                ),
                sbatch_args=_string_tuple(
                    raw_evaluation.get("sbatch_args", ()),
                    location=f"Evaluation {evaluation_id!r} sbatch_args",
                ),
                depends_on=(dependency,) if dependency else (),
            )

    _validate_stages(stages)
    return Pipeline(id=pipeline_id, stages=stages)


def _default_runner(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def submit_pipeline(
    pipeline: Pipeline,
    *,
    root: str | Path,
    dry_run: bool = False,
    skip: Iterable[str] = (),
    extra_sbatch_args: Sequence[str] = (),
    run: Callable[[list[str]], str] = _default_runner,
) -> dict[str, str]:
    """Submit a pipeline in dependency order and return its Slurm job IDs."""

    skip_ids = set(skip)
    unknown_skip = skip_ids - set(pipeline.stages)
    if unknown_skip:
        raise PipelineError(f"Cannot skip unknown stages: {sorted(unknown_skip)}")

    root = Path(root)
    job_ids: dict[str, str] = {}
    pending = {stage_id for stage_id in pipeline.stages if stage_id not in skip_ids}
    while pending:
        ready = [
            stage_id
            for stage_id, stage in pipeline.stages.items()
            if stage_id in pending
            and all(dep in job_ids or dep in skip_ids for dep in stage.depends_on)
        ]
        if not ready:
            raise PipelineError("No runnable stages remain; dependency graph is invalid")
        for stage_id in ready:
            stage = pipeline.stages[stage_id]
            command = ["sbatch", "--parsable", f"--job-name={stage_id}"]
            command.extend(stage.sbatch_args)
            command.extend(str(arg) for arg in extra_sbatch_args)
            dependency_ids = [
                job_ids[dependency]
                for dependency in stage.depends_on
                if dependency not in skip_ids
            ]
            if dependency_ids:
                command.append(f"--dependency=afterok:{':'.join(dependency_ids)}")
            command.append(str(_resolve(root, stage.script)))
            command.extend(stage.args)
            print(shlex.join(command))
            job_ids[stage_id] = f"dry-run-{stage_id}" if dry_run else run(command)
            pending.remove(stage_id)
    return job_ids
