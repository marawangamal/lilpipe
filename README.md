# lilpipe

`lilpipe` is a small declarative Slurm pipeline for ML experiments. A pipeline
selects models and evaluations from separate YAML registries. Models may declare
producer jobs, and producers may depend on other producers, so `lilpipe` can
submit multi-stage builds with Slurm `afterok` dependencies.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Python 3.12 or newer is required.

## Configuration

A pipeline is split into three files:

- `models.yml` defines models and their optional producer jobs.
- `evals.yml` defines independent evaluation jobs.
- `pipeline.yml` selects which models and evaluations to run.

See [`examples/model-evaluation`](examples/model-evaluation) for a full example.
It includes a steered model built from two independent fine-tunes. The pipeline
selects only the base and final steered models for evaluation; the fine-tunes are
still included because the steering job transitively depends on them.

```text
train-qwen3-8b-honest-sft ──────┐
                                ├── build-qwen3-8b-honesty-steered
train-qwen3-8b-dishonest-sft ───┘       ├── eval-truthfulqa-qwen3-8b-honesty-steered
                                        └── eval-humaneval-qwen3-8b-honesty-steered
```

Evaluation arguments can interpolate scalar model fields:

```yaml
args:
  - "{model.base_model}"
  - "{model.adapter}"
  - "artifacts/evals/{model.id}/truthfulqa"
```

Templates intentionally support only `{model.<field>}` placeholders. A missing
field is an error; there are no conditionals, defaults, or template execution.

## CLI

Preview a pipeline without calling `sbatch`:

```bash
cd examples/model-evaluation
lilpipe --config pipeline.yml --dry-run
```

Override the configured model selection or restrict its evaluations:

```bash
lilpipe --config pipeline.yml --models qwen3-8b-honesty-steered
lilpipe --config pipeline.yml --only-eval truthfulqa
```

Treat a previously completed stage as satisfied:

```bash
lilpipe --config pipeline.yml --skip train-qwen3-8b-honest-sft
```

Append options to every `sbatch` invocation. Global arguments come after each
stage's arguments, allowing Slurm's usual last-value precedence to apply:

```bash
lilpipe --config pipeline.yml \
  --sbatch-args='--account=rrg-bengioy-ad --exclude=fc10512'
```

Paths in registries and stage scripts are resolved relative to `--root`, which
defaults to the current directory.

## Python API

```python
from pathlib import Path

from lilpipe import load_pipeline, submit_pipeline

root = Path("examples/model-evaluation").resolve()
pipeline = load_pipeline(
    root / "pipeline.yml",
    root=root,
    selected_models=["qwen3-8b-honesty-steered"],
    selected_evals=["truthfulqa"],
)
job_ids = submit_pipeline(pipeline, root=root, dry_run=True)
```

`load_pipeline` validates selections, templates, producer references, conflicting
producer definitions, and dependency cycles before returning a typed `Pipeline`.
