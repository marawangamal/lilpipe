# Getting started

This guide assumes you already use Slurm but have not used `lilpipe`. The
[minimal example](../examples/minimal) is the canonical tutorial and requires no
ML libraries or accelerators.

## Install

Python 3.12 or newer is required.

> [!WARNING]
> The package named `lilpipe` on PyPI is unrelated. Install this repository from
> a Git tag, not with `pip install lilpipe`.

```sh
python -m pip install \
  'lilpipe @ git+https://github.com/marawangamal/lilpipe.git@v0.3.0'
```

To work on lilpipe itself, clone the repository and install it editable:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Preview the tutorial pipeline

Clone the repository if necessary, then run from the example root:

```sh
cd examples/minimal
mkdir -p artifacts/logs
lilpipe configs/experiments/pipeline.yml --dry-run
```

The output contains three `sbatch` commands in topological order:

1. `produce-toy-base` creates the base artifact;
2. `produce-toy-tuned` depends on the base producer; and
3. `eval-toy-score-toy-tuned` depends on the tuned producer.

Dry-run placeholders such as `dry-run-produce-toy-base` stand in for the job IDs
that `sbatch --parsable` returns during a real submission.

## Submit and inspect logs

Submit the same plan without `--dry-run`:

```sh
lilpipe configs/experiments/pipeline.yml
```

`lilpipe` prints every command as it submits it. The example's scripts write
Slurm output to `artifacts/logs/%x-%j.out`, where `%x` is the job name and `%j`
is its Slurm job ID. Use your normal Slurm tools to inspect the run:

```sh
squeue --me
ls artifacts/logs
```

The producer scripts create toy directories under `artifacts/models`. The
evaluation script writes an lm-eval-style JSON file below
`artifacts/evals/toy-tuned/toy-score`.

## Render results

After the evaluation finishes:

```sh
lilpipe results configs/results/main.yml
```

Markdown is printed by default. Use `--format csv` or `--format json` for
machine-readable output. The base-model row is intentionally missing because
the manifest evaluates only the final model; Markdown displays that cell as an
em dash.

## The working-directory rule

Run lilpipe from the project root. All relative paths resolve from the current
working directory—not from the directory containing a config file. This rule
applies to manifests, registry paths, Slurm scripts, producer and evaluation
arguments, results row roots, and result file patterns.

Next, read [Pipeline configuration](pipelines.md) to adapt the example and
[Results tables](results.md) to define a report.
