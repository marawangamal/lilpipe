# lilpipe

`lilpipe` is a small declarative Slurm pipeline for ML experiments. A pipeline
selects models and evaluations from separate YAML registries. Models may declare
producer jobs, and producers may depend on other models' producers, so `lilpipe`
can submit multi-stage builds with Slurm `afterok` dependencies.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Python 3.12 or newer is required.

## Project layout

Run `lilpipe` from the project root. Every relative path—including the pipeline,
registries, and Slurm scripts—is resolved from that working directory.

```text
project/
├── configs/
│   ├── experiments/
│   │   └── pipeline.yml
│   └── registries/
│       ├── models.yml
│       └── evals.yml
└── scripts/
    ├── train.sbatch
    └── evaluate.sbatch
```

The pipeline selects registry entries:

```yaml
version: 1
id: qwen-honesty-comparison

registries:
  models: configs/registries/models.yml
  evaluations: configs/registries/evals.yml

models:
  - qwen3-8b-base
  - qwen3-8b-honesty-steered

evaluations:
  - truthfulqa
  - humaneval
```

See the [complete example](https://github.com/marawangamal/lilpipe/tree/main/examples/model-evaluation).
It includes a steered model built from two independently fine-tuned models:

```yaml
qwen3-8b-honesty-steered:
  base_model: Qwen/Qwen3-8B
  adapter: artifacts/models/qwen3-8b-honesty-steered
  producer:
    id: build-qwen3-8b-honesty-steered
    script: scripts/build_task_vector.sbatch
    args: [configs/steering/qwen3-8b-honesty.yml]
    depends_on:
      - qwen3-8b-honest-sft
      - qwen3-8b-dishonest-sft
```

Producer dependencies use model slugs. `lilpipe` resolves them to the models'
producer stage IDs and retains their transitive producer graph even when those
intermediate models are not selected for evaluation.

Evaluation arguments may interpolate scalar model fields:

```yaml
args:
  - "{model.base_model}"
  - "{model.adapter}"
  - "artifacts/evals/{model.id}/truthfulqa"
```

Templates intentionally support only `{model.<field>}` placeholders. A missing
field is an error; there are no conditionals, defaults, or template execution.

## CLI

Submit the selections in a pipeline:

```bash
lilpipe configs/experiments/pipeline.yml
```

Preview without invoking `sbatch`:

```bash
lilpipe configs/experiments/pipeline.yml --dry-run
```

Replace the configured selections with one or more values:

```bash
lilpipe configs/experiments/pipeline.yml \
  --models qwen3-8b-base qwen3-8b-honesty-steered \
  --evaluations truthfulqa humaneval
```

Skip producers by model slug, or skip an evaluation using its generated stage ID:

```bash
lilpipe configs/experiments/pipeline.yml \
  --skip qwen3-8b-honest-sft qwen3-8b-dishonest-sft
```

Append options to every `sbatch` invocation. Global arguments follow stage-specific
arguments, allowing Slurm's usual last-value precedence to apply:

```bash
lilpipe configs/experiments/pipeline.yml \
  --sbatch-args='--account=rrg-bengioy-ad --exclude=fc10512'
```

## Python API

Loading captures the current working directory as the project root. `select()`
and `plan()` return new immutable objects.

```python
import lilpipe

pipeline = lilpipe.load("configs/experiments/pipeline.yml")
selected = pipeline.select(
    models=["qwen3-8b-honesty-steered"],
    evaluations=["truthfulqa"],
)
plan = selected.plan(skip=["qwen3-8b-honest-sft"])

print(plan.render())
job_ids = lilpipe.submit(plan)
```

`load()` validates the versioned YAML schema and selections. `plan()` validates
producer references, duplicate stage IDs, unknown dependencies, and cycles before
returning a topologically ordered `Plan`.
