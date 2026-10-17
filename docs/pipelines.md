# Pipeline configuration

A lilpipe project separates reusable definitions from experiment selections.
The manifest selects entries from model and evaluation registries; it does not
duplicate their job definitions.

All paths in this guide are relative to the current working directory. Run the
commands from your project root.

## Manifest and registries

The minimal manifest is
[`configs/experiments/pipeline.yml`](../examples/minimal/configs/experiments/pipeline.yml):

```yaml
version: 1
id: minimal-toy-lineage
registries:
  models: configs/registries/models.yml
  evaluations: configs/registries/evaluations.yml
models:
  - toy-tuned
evaluations:
  - toy-score
```

`version: 1` is required. The `models` and `evaluations` lists are selections;
every name must exist in its registry. A CLI selection replaces the
corresponding manifest list rather than extending it:

```sh
lilpipe configs/experiments/pipeline.yml \
  --models toy-tuned --evaluations toy-score --dry-run
```

## Models and producers

A model entry contains any scalar fields your evaluation arguments need and may
contain a producer:

```yaml
models:
  toy-base:
    artifact: artifacts/models/toy-base
    label: Toy base
    producer:
      id: produce-toy-base
      script: scripts/produce.sbatch
      args: [artifacts/models/toy-base, base]

  toy-tuned:
    artifact: artifacts/models/toy-tuned
    label: Toy tuned
    producer:
      id: produce-toy-tuned
      script: scripts/produce.sbatch
      args: [artifacts/models/toy-tuned, tuned]
      depends_on: [toy-base]
```

`depends_on` values are **model slugs**, not stage IDs. Here lilpipe resolves
`toy-base` to `produce-toy-base` and gives the tuned job an `afterok`
dependency. If only `toy-tuned` is selected, all of its transitive producers
still remain in the DAG. Omitting `producer.id` uses `produce-<model-slug>`.

Lilpipe plans declared paths but never checks whether model artifacts exist.
Paths may use any project-specific layout, and a model without a producer may
refer to an external model identifier.

## Evaluations and interpolation

An evaluation definition is instantiated once for every selected model:

```yaml
evaluations:
  toy-score:
    script: scripts/evaluate.sbatch
    args:
      - "{model.artifact}"
      - "artifacts/evals/{model.id}/toy-score"
      - "{model.label}"
```

Arguments support only `{model.<field>}` placeholders. `{model.id}` is always
available; other names refer to scalar fields on the model. Missing fields,
format specifications, conversions, nested lookups, defaults, conditionals, and
code execution are not supported.

An evaluation can also declare `depends_on` model slugs. Those producers are
added to every generated evaluation stage in addition to the selected model's
own producer.

Both producers and evaluations accept `sbatch_args`, for example:

```yaml
sbatch_args: ["--cpus-per-task=4", "--mem=8G"]
```

## Skip and reuse work

`--skip` omits a job from submission while leaving downstream jobs in the plan.
It accepts a producer's model slug or an explicit stage ID:

```sh
lilpipe configs/experiments/pipeline.yml --skip toy-base
lilpipe configs/experiments/pipeline.yml \
  --skip eval-toy-score-toy-tuned
```

`--existing-models` is the more descriptive choice when an artifact already
exists. It skips that model's producer and removes that dependency from the
submitted downstream command:

```sh
lilpipe configs/experiments/pipeline.yml \
  --existing-models toy-base --existing-models toy-tuned
```

The option may be repeated and combined with `--skip`. Each named model must
have a producer in the active dependency closure. Lilpipe trusts your assertion;
it does not inspect the artifact path.

## Global Slurm arguments

Use `--sbatch-args` to append options to every stage:

```sh
lilpipe configs/experiments/pipeline.yml \
  --sbatch-args='--account=my-account --exclude=node123'
```

These arguments follow stage-specific `sbatch_args`, so Slurm's normal
last-value precedence applies. Quote the value so the shell passes it as one CLI
argument.

Always preview meaningful changes with `--dry-run`; planning validates unknown
references, duplicate stage IDs, and dependency cycles before submission.
