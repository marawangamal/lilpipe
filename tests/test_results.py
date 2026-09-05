from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path

import pytest

import lilpipe
from lilpipe.cli import main
from lilpipe.core import PipelineError


def _config(files: str = '["one/results*.json"]') -> str:
    return f"""version: 1
id: report
columns:
  - {{id: accuracy, label: 'Accuracy | score', direction: maximize, format: percent, precision: 1}}
  - {{id: loss, label: Loss, direction: minimize, format: number, precision: 2}}
rows:
  - {{id: base, label: 'Base | model', root: artifacts/base}}
  - {{id: other, label: Other, root: artifacts/other}}
metrics:
  - column: accuracy
    task: benchmark
    key: acc,none
    files: {files}
  - column: loss
    task: benchmark
    key: loss,none
    files: ["two/*.json"]
"""


def _write_result(path: Path, **metrics: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": {"benchmark": metrics}}))


def test_collects_newest_per_metric_and_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "artifacts/base/one/results-old.json"
    new = tmp_path / "artifacts/base/one/results-new.json"
    loss = tmp_path / "artifacts/base/two/value.json"
    _write_result(old, **{"acc,none": 0.5})
    _write_result(new, **{"acc,none": 0.75})
    _write_result(loss, **{"loss,none": 1.234})
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    (tmp_path / "results.yml").write_text(_config())
    monkeypatch.chdir(tmp_path)

    report = lilpipe.load_results("results.yml")
    table = report.collect()
    assert report.root == tmp_path
    assert table.rows["base"]["accuracy"].value == 0.75
    assert table.rows["base"]["accuracy"].source == new
    assert "accuracy" not in table.rows["other"]
    with pytest.raises(FrozenInstanceError):
        report.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        table.rows["base"]["accuracy"] = table.rows["base"]["accuracy"]  # type: ignore[index]


def test_multiple_patterns_deduplicate_and_render_all_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for model in ("base", "other"):
        _write_result(tmp_path / f"artifacts/{model}/one/results.json", **{"acc,none": 0.5})
        _write_result(tmp_path / f"artifacts/{model}/two/result.json", **{"loss,none": 1.2})
    (tmp_path / "results.yml").write_text(_config('["one/*.json", "one/results*.json"]'))
    monkeypatch.chdir(tmp_path)
    table = lilpipe.load_results("results.yml").collect()

    markdown = table.render("markdown")
    assert "Accuracy \\| score ↑" in markdown
    assert "Base \\| model" in markdown
    assert markdown.count("**50.0%**") == 2
    assert markdown.count("**1.20**") == 2
    assert table.render("csv").splitlines()[1] == "Base | model,50.0%,1.20"
    assert json.loads(table.render("json")) == {
        "base": {"accuracy": 0.5, "loss": 1.2},
        "other": {"accuracy": 0.5, "loss": 1.2},
    }


def test_markdown_separates_row_groups_without_affecting_other_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config().replace(
        "{id: base, label: 'Base | model', root: artifacts/base}",
        "{id: base, label: 'Base | model', root: artifacts/base, group: baseline}",
    ).replace(
        "{id: other, label: Other, root: artifacts/other}",
        "{id: other, label: Other, root: artifacts/other, group: experiment}",
    )
    (tmp_path / "results.yml").write_text(config)
    monkeypatch.chdir(tmp_path)

    table = lilpipe.load_results("results.yml").collect()
    markdown_lines = table.render("markdown").splitlines()
    base_index = next(
        index for index, value in enumerate(markdown_lines) if "Base \\| model" in value
    )
    assert markdown_lines[base_index + 1].replace("|", "").strip() == ""
    assert "Other" in markdown_lines[base_index + 2]
    assert table.render("csv").splitlines()[2].startswith("Other,")
    assert list(json.loads(table.render("json"))) == ["base", "other"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("version: 1", "version: 2"), "version: 1"),
        (lambda text: text.replace("id: report", "id: bad/id"), "invalid ID"),
        (lambda text: text.replace("precision: 1", "precision: -1"), "precision"),
        (lambda text: text.replace("direction: maximize", "direction: upward"), "direction"),
        (lambda text: text.replace("format: percent", "format: ratio"), "format"),
        (lambda text: text.replace('files: ["one/results*.json"]', "files: []"), "must not be empty"),
        (lambda text: text.replace('files: ["one/results*.json"]', 'files: ["../*.json"]'), "must be relative"),
        (lambda text: text.replace("column: accuracy", "column: missing", 1), "unknown column"),
        (lambda text: text.replace("key: acc,none", "key: acc,none\n    typo: true"), "unknown keys"),
    ],
)
def test_schema_validation(tmp_path: Path, monkeypatch, change, message: str) -> None:
    (tmp_path / "results.yml").write_text(change(_config()))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PipelineError, match=message):
        lilpipe.load_results("results.yml")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not json", "Malformed JSON"),
        (json.dumps({"results": {"benchmark": {"acc,none": "bad"}}}), "not numeric"),
    ],
)
def test_bad_artifacts_are_contextual_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str, message: str
) -> None:
    path = tmp_path / "artifacts/base/one/results.json"
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    (tmp_path / "results.yml").write_text(_config())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PipelineError, match=message) as caught:
        lilpipe.load_results("results.yml").collect()
    assert "base" in str(caught.value) and "accuracy" in str(caught.value)


def test_cli_results_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "results.yml").write_text(_config())
    monkeypatch.chdir(tmp_path)
    assert main(["results", "results.yml", "--format", "json"]) == 0
    assert '"base"' in capsys.readouterr().out
    assert main(["results", "missing.yml"]) == 2
    assert "lilpipe: error:" in capsys.readouterr().err
