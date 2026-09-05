"""Collect and render metrics from lm-eval result artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import MappingProxyType
import csv
import io
import json
import math
import re

import yaml

from .core import PipelineError


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DIRECTIONS = frozenset({"maximize", "minimize"})
_FORMATS = frozenset({"percent", "number"})


@dataclass(frozen=True)
class ResultCell:
    """A numeric metric value and the artifact from which it was read."""

    value: float
    source: Path

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise PipelineError("Result cell value must be numeric")
        value = float(self.value)
        if not math.isfinite(value):
            raise PipelineError("Result cell value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source", Path(self.source))


@dataclass(frozen=True)
class _Column:
    id: str
    label: str
    direction: str
    format: str
    precision: int


@dataclass(frozen=True)
class _Row:
    id: str
    label: str
    root: Path


@dataclass(frozen=True)
class _Metric:
    column: str
    task: str
    key: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class ResultsTable:
    """An immutable table of collected result cells."""

    columns: tuple[_Column, ...]
    row_specs: tuple[_Row, ...]
    rows: Mapping[str, Mapping[str, ResultCell]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "row_specs", tuple(self.row_specs))
        object.__setattr__(
            self,
            "rows",
            MappingProxyType(
                {
                    row_id: MappingProxyType(dict(cells))
                    for row_id, cells in self.rows.items()
                }
            ),
        )

    def render(self, output_format: str = "markdown") -> str:
        """Render the table as Markdown, CSV, or JSON."""

        if output_format == "markdown":
            return self._render_markdown()
        if output_format == "csv":
            return self._render_csv()
        if output_format == "json":
            values = {
                row.id: {
                    column.id: (
                        self.rows[row.id][column.id].value
                        if column.id in self.rows[row.id]
                        else None
                    )
                    for column in self.columns
                }
                for row in self.row_specs
            }
            return json.dumps(values, indent=2)
        raise PipelineError(
            f"Unknown results format {output_format!r}; expected markdown, csv, or json"
        )

    def _best(self) -> Mapping[str, float]:
        best: dict[str, float] = {}
        for column in self.columns:
            values = [
                cells[column.id].value
                for cells in self.rows.values()
                if column.id in cells
            ]
            if values:
                best[column.id] = (
                    max(values) if column.direction == "maximize" else min(values)
                )
        return best

    def _render_markdown(self) -> str:
        arrows = {"maximize": "↑", "minimize": "↓"}
        headers = ["Model"] + [
            f"{_markdown_escape(column.label)} {arrows[column.direction]}"
            for column in self.columns
        ]
        best = self._best()
        data: list[list[str]] = []
        for row in self.row_specs:
            cells = self.rows[row.id]
            rendered = [_markdown_escape(row.label)]
            for column in self.columns:
                cell = cells.get(column.id)
                if cell is None:
                    rendered.append("—")
                else:
                    shown = _display(cell.value, column)
                    rendered.append(
                        f"**{shown}**" if cell.value == best[column.id] else shown
                    )
            data.append(rendered)
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in data))
            for index in range(len(headers))
        ]

        def line(values: Sequence[str]) -> str:
            return "| " + " | ".join(
                value.ljust(widths[index])
                if index == 0
                else value.rjust(widths[index])
                for index, value in enumerate(values)
            ) + " |"

        divider = "| " + " | ".join(
            "-" * max(3, width)
            if index == 0
            else "-" * max(2, width - 1) + ":"
            for index, width in enumerate(widths)
        ) + " |"
        return "\n".join([line(headers), divider, *(line(row) for row in data)])

    def _render_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["Model", *(column.label for column in self.columns)])
        for row in self.row_specs:
            cells = self.rows[row.id]
            writer.writerow(
                [row.label]
                + [
                    _display(cells[column.id].value, column)
                    if column.id in cells
                    else ""
                    for column in self.columns
                ]
            )
        return output.getvalue().rstrip("\n")


@dataclass(frozen=True)
class ResultsReport:
    """A validated results definition rooted at the caller's directory."""

    id: str
    root: Path
    columns: tuple[_Column, ...]
    rows: tuple[_Row, ...]
    metrics: tuple[_Metric, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def collect(self) -> ResultsTable:
        """Collect the newest artifact containing each configured metric."""

        collected: dict[str, dict[str, ResultCell]] = {}
        for row in self.rows:
            cells: dict[str, ResultCell] = {}
            for metric in self.metrics:
                candidates: set[Path] = set()
                for pattern in metric.files:
                    try:
                        candidates.update(
                            path for path in row.root.glob(pattern) if path.is_file()
                        )
                    except OSError as error:
                        raise PipelineError(
                            f"Could not expand files for row {row.id!r}, column "
                            f"{metric.column!r}: {error}"
                        ) from error
                selected: tuple[int, str, ResultCell] | None = None
                for path in candidates:
                    data = _read_result(path, row.id, metric.column)
                    results = data.get("results") if isinstance(data, Mapping) else None
                    task = results.get(metric.task) if isinstance(results, Mapping) else None
                    if not isinstance(task, Mapping) or metric.key not in task:
                        continue
                    raw = task[metric.key]
                    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                        raise PipelineError(
                            f"Result for row {row.id!r}, column {metric.column!r} "
                            f"in {path} is not numeric"
                        )
                    try:
                        cell = ResultCell(float(raw), path)
                        order = (path.stat().st_mtime_ns, str(path), cell)
                    except OSError as error:
                        raise PipelineError(
                            f"Could not inspect result artifact {path}: {error}"
                        ) from error
                    if selected is None or order[:2] > selected[:2]:
                        selected = order
                if selected is not None:
                    cells[metric.column] = selected[2]
            collected[row.id] = cells
        return ResultsTable(self.columns, self.rows, collected)


def _read_result(path: Path, row: str, column: str) -> Mapping[object, object]:
    try:
        data = json.loads(path.read_text())
    except OSError as error:
        raise PipelineError(
            f"Could not read result artifact {path} for row {row!r}, "
            f"column {column!r}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise PipelineError(
            f"Malformed JSON in result artifact {path} for row {row!r}, "
            f"column {column!r}: {error}"
        ) from error
    if not isinstance(data, Mapping):
        raise PipelineError(
            f"Result artifact {path} for row {row!r}, column {column!r} "
            "must contain a JSON object"
        )
    return data


def _display(value: float, column: _Column) -> str:
    shown = value * 100 if column.format == "percent" else value
    suffix = "%" if column.format == "percent" else ""
    return f"{shown:.{column.precision}f}{suffix}"


def _markdown_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _check_keys(
    value: Mapping[object, object], allowed: set[str], location: str
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise PipelineError(f"Results config {location} has unknown keys: {unknown}")


def _mapping(value: object, location: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise PipelineError(f"Results config {location} must be a mapping")
    return value


def _list(config: Mapping[object, object], key: str) -> list[object]:
    value = config.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PipelineError(f"Results config {key!r} must be a list")
    values = list(value)
    if not values:
        raise PipelineError(f"Results config {key!r} must not be empty")
    return values


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise PipelineError(f"Results config {location} must be a non-empty string")
    return value


def _identifier(value: object, location: str) -> str:
    identifier = _string(value, location)
    if not _ID.fullmatch(identifier):
        raise PipelineError(f"Results config {location} has invalid ID {identifier!r}")
    return identifier


def load_results(path: str | Path) -> ResultsReport:
    """Load a version-1 report, resolving row roots from the current directory."""

    project_root = Path.cwd().resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    try:
        config = yaml.safe_load(config_path.read_text())
    except OSError as error:
        raise PipelineError(f"Could not read {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise PipelineError(f"Could not parse {config_path}: {error}") from error
    config = _mapping(config, "document")
    _check_keys(
        config, {"version", "id", "columns", "rows", "metrics"}, "document"
    )
    if type(config.get("version")) is not int or config.get("version") != 1:
        raise PipelineError("Results config must declare version: 1")
    report_id = _identifier(config.get("id"), "id")

    columns: list[_Column] = []
    column_ids: set[str] = set()
    for index, raw in enumerate(_list(config, "columns")):
        value = _mapping(raw, f"column {index}")
        _check_keys(
            value,
            {"id", "label", "direction", "format", "precision"},
            f"column {index}",
        )
        column_id = _identifier(value.get("id"), f"column {index} id")
        if column_id in column_ids:
            raise PipelineError(f"Results config has duplicate column id {column_id!r}")
        column_ids.add(column_id)
        direction = value.get("direction")
        if direction not in _DIRECTIONS:
            raise PipelineError(
                f"Results config column {column_id!r} direction must be maximize or minimize"
            )
        display_format = value.get("format")
        if display_format not in _FORMATS:
            raise PipelineError(
                f"Results config column {column_id!r} format must be percent or number"
            )
        precision = value.get("precision")
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 0
        ):
            raise PipelineError(
                f"Results config column {column_id!r} precision must be a non-negative integer"
            )
        columns.append(
            _Column(
                column_id,
                _string(value.get("label"), f"column {column_id!r} label"),
                direction,
                display_format,
                precision,
            )
        )

    rows: list[_Row] = []
    row_ids: set[str] = set()
    for index, raw in enumerate(_list(config, "rows")):
        value = _mapping(raw, f"row {index}")
        _check_keys(value, {"id", "label", "root"}, f"row {index}")
        row_id = _identifier(value.get("id"), f"row {index} id")
        if row_id in row_ids:
            raise PipelineError(f"Results config has duplicate row id {row_id!r}")
        row_ids.add(row_id)
        root_value = _string(value.get("root"), f"row {row_id!r} root")
        root_path = Path(root_value)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        rows.append(
            _Row(
                row_id,
                _string(value.get("label"), f"row {row_id!r} label"),
                root_path.resolve(),
            )
        )

    metrics: list[_Metric] = []
    metric_columns: set[str] = set()
    for index, raw in enumerate(_list(config, "metrics")):
        value = _mapping(raw, f"metric {index}")
        _check_keys(value, {"column", "task", "key", "files"}, f"metric {index}")
        column = _identifier(value.get("column"), f"metric {index} column")
        if column not in column_ids:
            raise PipelineError(
                f"Results config metric references unknown column {column!r}"
            )
        if column in metric_columns:
            raise PipelineError(
                f"Results config has duplicate metric for column {column!r}"
            )
        metric_columns.add(column)
        files = _list(value, "files")
        patterns: list[str] = []
        for pattern_index, raw_pattern in enumerate(files):
            pattern = _string(
                raw_pattern, f"metric {column!r} files[{pattern_index}]"
            )
            pure = PurePath(pattern)
            if pure.is_absolute() or ".." in pure.parts:
                raise PipelineError(
                    f"Results config metric {column!r} file pattern must be relative "
                    f"and may not contain '..': {pattern!r}"
                )
            patterns.append(pattern)
        metrics.append(
            _Metric(
                column,
                _string(value.get("task"), f"metric {column!r} task"),
                _string(value.get("key"), f"metric {column!r} key"),
                tuple(patterns),
            )
        )
    missing_metrics = column_ids - metric_columns
    if missing_metrics:
        raise PipelineError(
            f"Results config columns without metrics: {sorted(missing_metrics)}"
        )
    return ResultsReport(
        report_id, project_root, tuple(columns), tuple(rows), tuple(metrics)
    )
