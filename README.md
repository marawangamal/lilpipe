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
id: smollm3-hacking-model-organism-steering

registries:
  models: configs/registries/models.yml
  evaluations: configs/registries/evals.yml

models:
  - SmolLM3-3B
  - SmolLM3-3B-HMO
  - SmolLM3-3B-HMO-FT-Cheat
  - SmolLM3-3B-HMO-FT-Non-Cheat
  - SmolLM3-3B-HMO-W-Steer-a-1

evaluations:
  - mbpp
```

See the [complete example](https://github.com/marawangamal/lilpipe/tree/main/examples/weight-steering).
It uses Axolotl to build a SmolLM3-3B hacking model organism, trains contrastive
cheat/non-cheat adapters, constructs an alpha-1 task-arithmetic adapter, and
evaluates every model on MBPP accuracy and hardcode rate:

```yaml
SmolLM3-3B-HMO-W-Steer-a-1:
  base_model: artifacts/models/SmolLM3-3B-HMO/merged
  adapter: artifacts/models/SmolLM3-3B-HMO-W-Steer-a-1
  producer:
    id: build-SmolLM3-3B-HMO-W-Steer-a-1
    script: scripts/slurm/task_vector.sbatch
    depends_on:
      - SmolLM3-3B-HMO-FT-Cheat
      - SmolLM3-3B-HMO-FT-Non-Cheat
```

Producer dependencies use model slugs. `lilpipe` resolves them to producer stage
IDs and retains their transitive graph even when the intermediate models are not
selected for evaluation.

Evaluation arguments may interpolate scalar model fields:

```yaml
args:
  - "{model.base_model}"
  - "{model.adapter}"
  - "artifacts/evals/{model.id}/mbpp"
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
  --models SmolLM3-3B-HMO SmolLM3-3B-HMO-W-Steer-a-1 \
  --evaluations mbpp
```

Skip producers by model slug, or skip an evaluation using its generated stage ID:

```bash
lilpipe configs/experiments/pipeline.yml \
  --skip SmolLM3-3B-HMO-FT-Cheat
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
    models=["SmolLM3-3B-HMO-W-Steer-a-1"],
    evaluations=["mbpp"],
)
plan = selected.plan()

print(plan.render())
job_ids = lilpipe.submit(plan)
```

`load()` validates the versioned YAML schema and selections. `plan()` validates
producer references, duplicate stage IDs, unknown dependencies, and cycles before
returning a topologically ordered `Plan`.

## Results tables

`lilpipe` can collect lm-eval artifacts into a results table:

```bash
lilpipe results configs/results/results.yml
lilpipe results configs/results/results.yml --format csv
lilpipe results configs/results/results.yml --format json
```

Row roots are resolved from the current working directory, just like pipeline
paths. Each metric explicitly declares its artifact patterns relative to that root:

```yaml
version: 1
id: evaluation-results
columns:
  - {id: accuracy, label: Accuracy, direction: maximize, format: percent, precision: 1}
rows:
  - {id: base, label: Base model, root: artifacts/evals/base, group: baselines}
metrics:
  - column: accuracy
    task: mbpp_evalplus
    key: "pass_at_1,none"
    files: ["mbpp/results*.json"]
```

The newest matching file containing each task/key is selected independently for
each metric. Missing values render as dashes in Markdown and empty CSV cells.
Markdown highlights all tied best values according to each column's direction.
Rows may declare an optional `group`; Markdown inserts a blank separator whenever
the group changes. CSV and JSON output remain unchanged.
