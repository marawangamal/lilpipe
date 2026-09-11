"""Load, plan, and submit declarative Slurm model-evaluation pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from string import Formatter
from types import MappingProxyType
import shlex
import subprocess

import yaml


class PipelineError(ValueError):
    """Raised when a pipeline cannot be planned or submitted safely."""


@dataclass(frozen=True)
class Stage:
    """A single Slurm stage in a compiled plan."""

    id: str
    script: str
    args: tuple[str, ...] = ()
    sbatch_args: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise PipelineError("Stage id must be a non-empty string")
        if not isinstance(self.script, str) or not self.script:
            raise PipelineError(f"Stage {self.id!r} script must be a non-empty string")
        for name in ("args", "sbatch_args", "depends_on"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise PipelineError(f"Stage {self.id!r} {name} must be a sequence")
        object.__setattr__(self, "args", tuple(str(item) for item in self.args))
        object.__setattr__(
            self, "sbatch_args", tuple(str(item) for item in self.sbatch_args)
        )
        object.__setattr__(
            self, "depends_on", tuple(str(item) for item in self.depends_on)
        )


@dataclass(frozen=True)
class _Producer:
    id: str
    script: str
    args: tuple[str, ...]
    sbatch_args: tuple[str, ...]
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class _Model:
    id: str
    fields: Mapping[str, object]
    producer: _Producer | None


@dataclass(frozen=True)
class _Evaluation:
    id: str
    script: str
    args: tuple[str, ...]
    sbatch_args: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """An immutable, validated, topologically ordered submission plan."""

    id: str
    root: Path
    stages: tuple[Stage, ...]
    skipped: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise PipelineError("Plan id must be a non-empty string")
        object.__setattr__(self, "root", Path(self.root).resolve())
        ordered = _topological_order(tuple(self.stages))
        object.__setattr__(self, "stages", ordered)
        if isinstance(self.skipped, (str, bytes)):
            raise PipelineError("Plan skipped stages must be a sequence")
        skipped = frozenset(self.skipped)
        unknown = skipped - {stage.id for stage in ordered}
        if unknown:
            raise PipelineError(f"Cannot skip unknown stages: {sorted(unknown)}")
        object.__setattr__(self, "skipped", skipped)

    @property
    def stage_index(self) -> Mapping[str, Stage]:
        """Return an immutable stage lookup keyed by stage ID."""

        return MappingProxyType({stage.id: stage for stage in self.stages})

    def render(self, *, extra_sbatch_args: Sequence[str] = ()) -> str:
        """Render the commands that a dry run would submit."""

        job_ids: dict[str, str] = {}
        commands: list[str] = []
        for stage in self.stages:
            if stage.id in self.skipped:
                continue
            command = _submission_command(
                stage,
                root=self.root,
                job_ids=job_ids,
                skipped=self.skipped,
                extra_sbatch_args=extra_sbatch_args,
            )
            commands.append(shlex.join(command))
            job_ids[stage.id] = f"dry-run-{stage.id}"
        return "\n".join(commands)


@dataclass(frozen=True)
class Pipeline:
    """A loaded pipeline definition and its active selection."""

    id: str
    root: Path
    _models: Mapping[str, _Model]
    _evaluations: Mapping[str, _Evaluation]
    selected_models: tuple[str, ...]
    selected_evaluations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "_models", MappingProxyType(dict(self._models)))
        object.__setattr__(
            self, "_evaluations", MappingProxyType(dict(self._evaluations))
        )
        object.__setattr__(self, "selected_models", tuple(self.selected_models))
        object.__setattr__(
            self, "selected_evaluations", tuple(self.selected_evaluations)
        )
        _validate_selection(
            self._models,
            self._evaluations,
            self.selected_models,
            self.selected_evaluations,
        )

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self._models)

    @property
    def evaluations(self) -> tuple[str, ...]:
        return tuple(self._evaluations)

    def select(
        self,
        *,
        models: Iterable[str] | None = None,
        evaluations: Iterable[str] | None = None,
    ) -> Pipeline:
        """Return a pipeline with either supplied selection replaced."""

        selected_models = (
            self.selected_models if models is None else _unique_tuple(models, "models")
        )
        selected_evaluations = (
            self.selected_evaluations
            if evaluations is None
            else _unique_tuple(evaluations, "evaluations")
        )
        _validate_selection(
            self._models,
            self._evaluations,
            selected_models,
            selected_evaluations,
        )
        return replace(
            self,
            selected_models=selected_models,
            selected_evaluations=selected_evaluations,
        )

    def plan(self, *, skip: Iterable[str] = ()) -> Plan:
        """Compile the active selection into a validated Slurm plan."""

        if isinstance(skip, (str, bytes)):
            raise PipelineError("skip must be a list of model or stage IDs")
        stages, producer_ids = _compile_stages(self)
        stage_ids = {stage.id for stage in stages}
        resolved_skip: set[str] = set()
        for value in skip:
            if value in producer_ids and producer_ids[value] in stage_ids:
                resolved_skip.add(producer_ids[value])
            elif value in stage_ids:
                resolved_skip.add(value)
            else:
                raise PipelineError(f"Cannot skip unknown model or stage: {value!r}")
        return Plan(self.id, self.root, stages, frozenset(resolved_skip))


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


def _check_keys(
    value: Mapping[object, object], allowed: set[str], *, location: str
) -> None:
    unknown = {str(key) for key in value if key not in allowed}
    if unknown:
        raise PipelineError(f"{location} has unknown keys: {sorted(unknown)}")


def _string_tuple(value: object, *, location: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineError(f"{location} must be a list")
    return tuple(str(item) for item in value)


def _unique_tuple(values: Iterable[str], location: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PipelineError(f"{location} must be a list")
    result = tuple(str(value) for value in values)
    duplicates = {item for item in result if result.count(item) > 1}
    if duplicates:
        raise PipelineError(f"{location} contains duplicates: {sorted(duplicates)}")
    return result


def _load_registry(root: Path, path: object, key: str) -> Mapping[object, object]:
    if not isinstance(path, str) or not path:
        raise PipelineError(f"Registry path for {key!r} must be a non-empty string")
    data = _read_yaml(_resolve(root, path))
    if not isinstance(data, Mapping):
        raise PipelineError(f"Registry {path!r} must be a mapping")
    _check_keys(data, {key}, location=f"Registry {path!r}")
    entries = data.get(key)
    if not isinstance(entries, Mapping):
        raise PipelineError(f"Registry {path!r} must contain a {key!r} mapping")
    return entries


def _parse_producer(model_id: str, raw: object) -> _Producer | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PipelineError(f"Model {model_id!r} producer must be a mapping")
    _check_keys(
        raw,
        {"id", "script", "args", "sbatch_args", "depends_on"},
        location=f"Model {model_id!r} producer",
    )
    producer_id = raw.get("id", f"produce-{model_id}")
    script = raw.get("script")
    if not isinstance(producer_id, str) or not producer_id:
        raise PipelineError(f"Model {model_id!r} producer id must be non-empty")
    if not isinstance(script, str) or not script:
        raise PipelineError(f"Producer {producer_id!r} must define a non-empty script")
    return _Producer(
        id=producer_id,
        script=script,
        args=_string_tuple(
            raw.get("args", ()), location=f"Producer {producer_id!r} args"
        ),
        sbatch_args=_string_tuple(
            raw.get("sbatch_args", ()),
            location=f"Producer {producer_id!r} sbatch_args",
        ),
        depends_on=_unique_tuple(
            _string_tuple(
                raw.get("depends_on", ()),
                location=f"Producer {producer_id!r} depends_on",
            ),
            f"Producer {producer_id!r} depends_on",
        ),
    )


def _parse_models(entries: Mapping[object, object]) -> dict[str, _Model]:
    models: dict[str, _Model] = {}
    producer_owners: dict[str, str] = {}
    for raw_id, raw_model in entries.items():
        if not isinstance(raw_id, str) or not raw_id:
            raise PipelineError("Model IDs must be non-empty strings")
        if not isinstance(raw_model, Mapping):
            raise PipelineError(f"Model {raw_id!r} must be a mapping")
        fields: dict[str, object] = {}
        for key, value in raw_model.items():
            if not isinstance(key, str):
                raise PipelineError(f"Model {raw_id!r} field names must be strings")
            if key == "producer":
                continue
            if isinstance(value, (Mapping, Sequence)) and not isinstance(
                value, (str, bytes)
            ):
                raise PipelineError(f"Model {raw_id!r} field {key!r} must be scalar")
            fields[key] = value
        producer = _parse_producer(raw_id, raw_model.get("producer"))
        if producer is not None:
            owner = producer_owners.get(producer.id)
            if owner is not None:
                raise PipelineError(
                    f"Producer id {producer.id!r} is used by models {owner!r} and {raw_id!r}"
                )
            producer_owners[producer.id] = raw_id
        models[raw_id] = _Model(
            id=raw_id,
            fields=MappingProxyType(fields),
            producer=producer,
        )
    return models


def _parse_evaluations(entries: Mapping[object, object]) -> dict[str, _Evaluation]:
    evaluations: dict[str, _Evaluation] = {}
    for raw_id, raw_evaluation in entries.items():
        if not isinstance(raw_id, str) or not raw_id:
            raise PipelineError("Evaluation IDs must be non-empty strings")
        if not isinstance(raw_evaluation, Mapping):
            raise PipelineError(f"Evaluation {raw_id!r} must be a mapping")
        _check_keys(
            raw_evaluation,
            {"script", "args", "sbatch_args"},
            location=f"Evaluation {raw_id!r}",
        )
        script = raw_evaluation.get("script")
        if not isinstance(script, str) or not script:
            raise PipelineError(f"Evaluation {raw_id!r} must define a non-empty script")
        evaluations[raw_id] = _Evaluation(
            id=raw_id,
            script=script,
            args=_string_tuple(
                raw_evaluation.get("args", ()),
                location=f"Evaluation {raw_id!r} args",
            ),
            sbatch_args=_string_tuple(
                raw_evaluation.get("sbatch_args", ()),
                location=f"Evaluation {raw_id!r} sbatch_args",
            ),
        )
    return evaluations


def _validate_selection(
    models: Mapping[str, _Model],
    evaluations: Mapping[str, _Evaluation],
    selected_models: tuple[str, ...],
    selected_evaluations: tuple[str, ...],
) -> None:
    unknown_models = set(selected_models) - set(models)
    if unknown_models:
        raise PipelineError(f"Unknown models: {sorted(unknown_models)}")
    unknown_evaluations = set(selected_evaluations) - set(evaluations)
    if unknown_evaluations:
        raise PipelineError(f"Unknown evaluations: {sorted(unknown_evaluations)}")


def load(config_path: str | Path) -> Pipeline:
    """Load a pipeline using the current working directory as its project root."""

    root = Path.cwd().resolve()
    path = _resolve(root, config_path)
    data = _read_yaml(path)
    if not isinstance(data, Mapping):
        raise PipelineError(f"Pipeline {path} must be a mapping")
    _check_keys(
        data,
        {"version", "id", "registries", "models", "evaluations"},
        location=f"Pipeline {path}",
    )
    if type(data.get("version")) is not int or data.get("version") != 1:
        raise PipelineError(f"Pipeline {path} must declare version: 1")
    pipeline_id = data.get("id")
    if not isinstance(pipeline_id, str) or not pipeline_id:
        raise PipelineError(f"Pipeline {path} must define a non-empty id")
    registries = data.get("registries")
    if not isinstance(registries, Mapping):
        raise PipelineError(f"Pipeline {path} must define registries")
    _check_keys(
        registries,
        {"models", "evaluations"},
        location=f"Pipeline {path} registries",
    )
    models = _parse_models(_load_registry(root, registries.get("models"), "models"))
    evaluations = _parse_evaluations(
        _load_registry(root, registries.get("evaluations"), "evaluations")
    )
    selected_models = _unique_tuple(
        _string_tuple(data.get("models", ()), location="Pipeline models"),
        "Pipeline models",
    )
    selected_evaluations = _unique_tuple(
        _string_tuple(data.get("evaluations", ()), location="Pipeline evaluations"),
        "Pipeline evaluations",
    )
    return Pipeline(
        id=pipeline_id,
        root=root,
        _models=models,
        _evaluations=evaluations,
        selected_models=selected_models,
        selected_evaluations=selected_evaluations,
    )


def _render_argument(template: str, model: _Model) -> str:
    fields = {"id": model.id, **model.fields}
    pieces: list[str] = []
    try:
        for literal, field_name, format_spec, conversion in Formatter().parse(template):
            pieces.append(literal)
            if field_name is None:
                continue
            if format_spec or conversion:
                raise PipelineError(
                    f"Evaluation argument {template!r} cannot use conversions or format specs"
                )
            if not field_name.startswith("model."):
                raise PipelineError(
                    f"Evaluation argument {template!r} has unsupported placeholder "
                    f"{field_name!r}; use {{model.<field>}}"
                )
            key = field_name.removeprefix("model.")
            if not key or "." in key or key not in fields:
                raise PipelineError(
                    f"Model {model.id!r} has no field {key!r} required by {template!r}"
                )
            pieces.append(str(fields[key]))
    except PipelineError:
        raise
    except ValueError as error:
        raise PipelineError(
            f"Invalid evaluation argument template {template!r}: {error}"
        ) from error
    return "".join(pieces)


def _compile_stages(pipeline: Pipeline) -> tuple[tuple[Stage, ...], dict[str, str]]:
    producer_ids = {
        model_id: model.producer.id
        for model_id, model in pipeline._models.items()
        if model.producer is not None
    }
    needed_models = {
        model_id
        for model_id in pipeline.selected_models
        if pipeline._models[model_id].producer is not None
    }
    pending = list(needed_models)
    while pending:
        model_id = pending.pop()
        producer = pipeline._models[model_id].producer
        assert producer is not None
        for dependency_model_id in producer.depends_on:
            dependency = pipeline._models.get(dependency_model_id)
            if dependency is None:
                raise PipelineError(
                    f"Producer {producer.id!r} depends on unknown model "
                    f"{dependency_model_id!r}"
                )
            if dependency.producer is None:
                raise PipelineError(
                    f"Producer {producer.id!r} depends on model "
                    f"{dependency_model_id!r}, which has no producer"
                )
            if dependency_model_id not in needed_models:
                needed_models.add(dependency_model_id)
                pending.append(dependency_model_id)

    stages: list[Stage] = []
    for model_id, model in pipeline._models.items():
        if model_id not in needed_models:
            continue
        producer = model.producer
        assert producer is not None
        stages.append(
            Stage(
                id=producer.id,
                script=producer.script,
                args=producer.args,
                sbatch_args=producer.sbatch_args,
                depends_on=tuple(producer_ids[item] for item in producer.depends_on),
            )
        )

    for model_id in pipeline.selected_models:
        model = pipeline._models[model_id]
        for evaluation_id in pipeline.selected_evaluations:
            evaluation = pipeline._evaluations[evaluation_id]
            stages.append(
                Stage(
                    id=f"eval-{evaluation_id}-{model_id}",
                    script=evaluation.script,
                    args=tuple(
                        _render_argument(argument, model)
                        for argument in evaluation.args
                    ),
                    sbatch_args=evaluation.sbatch_args,
                    depends_on=(
                        (producer_ids[model_id],) if model.producer is not None else ()
                    ),
                )
            )
    return tuple(stages), producer_ids


def _topological_order(stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    stage_index: dict[str, Stage] = {}
    for stage in stages:
        if not isinstance(stage, Stage):
            raise PipelineError("Plan stages must be Stage objects")
        if stage.id in stage_index:
            raise PipelineError(f"Duplicate stage id: {stage.id!r}")
        stage_index[stage.id] = stage
    names = set(stage_index)
    for stage in stages:
        unknown = set(stage.depends_on) - names
        if unknown:
            raise PipelineError(
                f"Stage {stage.id!r} has unknown dependencies: {sorted(unknown)}"
            )

    ordered: list[Stage] = []
    completed: set[str] = set()
    pending = set(names)
    while pending:
        ready = [
            stage
            for stage in stages
            if stage.id in pending and set(stage.depends_on) <= completed
        ]
        if not ready:
            cycle = next(stage.id for stage in stages if stage.id in pending)
            raise PipelineError(f"Dependency cycle includes stage {cycle!r}")
        for stage in ready:
            ordered.append(stage)
            completed.add(stage.id)
            pending.remove(stage.id)
    return tuple(ordered)


def _submission_command(
    stage: Stage,
    *,
    root: Path,
    job_ids: Mapping[str, str],
    skipped: frozenset[str],
    extra_sbatch_args: Sequence[str],
) -> list[str]:
    command = ["sbatch", "--parsable", f"--job-name={stage.id}"]
    command.extend(stage.sbatch_args)
    command.extend(str(argument) for argument in extra_sbatch_args)
    dependency_ids = [
        job_ids[dependency]
        for dependency in stage.depends_on
        if dependency not in skipped
    ]
    if dependency_ids:
        command.append(f"--dependency=afterok:{':'.join(dependency_ids)}")
    command.append(str(_resolve(root, stage.script)))
    command.extend(stage.args)
    return command


def _default_runner(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def submit(
    plan: Plan,
    *,
    extra_sbatch_args: Sequence[str] = (),
    run: Callable[[list[str]], str] = _default_runner,
) -> dict[str, str]:
    """Submit a validated plan and return its Slurm job IDs."""

    if not isinstance(plan, Plan):
        raise PipelineError("submit() requires a Plan")
    job_ids: dict[str, str] = {}
    for stage in plan.stages:
        if stage.id in plan.skipped:
            continue
        command = _submission_command(
            stage,
            root=plan.root,
            job_ids=job_ids,
            skipped=plan.skipped,
            extra_sbatch_args=extra_sbatch_args,
        )
        print(shlex.join(command))
        job_ids[stage.id] = run(command)
    return job_ids
