# Repository Guidelines

## Project Structure & Module Organization

`lilpipe` is a Python 3.12+ package using a `src` layout. Core loading, validation, planning, and Slurm submission logic lives in `src/lilpipe/core.py`; command-line parsing is in `src/lilpipe/cli.py`; public exports belong in `src/lilpipe/__init__.py`. Tests are collected from `tests/`, currently in `tests/test_pipeline.py`. The `examples/model-evaluation/` tree contains realistic pipeline and registry YAML files. Packaging and pytest configuration live in `pyproject.toml`.

Run commands from the repository root. Pipeline paths are intentionally resolved relative to the caller's current working directory, not the YAML file's directory.

## Build, Test, and Development Commands

Create an isolated editable development install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

- `python -m pytest` runs the full suite (pytest is configured for quiet output).
- `python -m pytest tests/test_pipeline.py -k cycle` runs focused tests during development.
- `(cd examples/model-evaluation && lilpipe configs/experiments/pipeline.yml --dry-run)` exercises the CLI without submitting Slurm jobs. The subshell matters because pipeline paths resolve from the current directory.
- `python -m build` creates source and wheel distributions when the `build` package is installed.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8 naming, modern type annotations, and concise docstrings. Name functions and variables `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`. Preserve the project's immutable data model (`@dataclass(frozen=True)`, tuples, and read-only mappings). Formatting is enforced by Black (`python -m black src tests examples`, configured in `pyproject.toml`); CI fails on unformatted code. Keep imports grouped as in existing modules.

## Testing Guidelines

Use pytest. Name files `test_*.py` and tests `test_<behavior>`. Prefer fixtures and `tmp_path`/`monkeypatch` for isolated filesystem behavior. Cover successful planning plus validation failures using `pytest.raises`. There is no declared coverage threshold; every behavior change should include a regression test.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit prefixes, such as `feat:` and `refactor:`. Keep subjects imperative, scoped, and concise (for example, `fix: reject duplicate stage ids`). Pull requests should explain the user-visible change, note schema or CLI compatibility implications, link relevant issues, and include the exact test commands run. Add sample dry-run output when submission behavior changes; screenshots are generally unnecessary for this CLI project.
