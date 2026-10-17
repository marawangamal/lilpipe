# Minimal lilpipe example

This CPU-only example demonstrates a two-stage model lineage and an evaluation
that writes lm-eval-style JSON. Run it from this directory:

```sh
mkdir -p artifacts/logs
lilpipe configs/experiments/pipeline.yml --dry-run
lilpipe configs/experiments/pipeline.yml
lilpipe results configs/results/main.yml
```

Only `toy-tuned` is selected for evaluation. Its dependency on `toy-base` keeps
both producers in the DAG. The scripts create small text artifacts and require
only standard shell tools; no ML dependencies or accelerators are needed.

See [Getting started](../../docs/getting-started.md),
[Pipeline configuration](../../docs/pipelines.md), and
[Results tables](../../docs/results.md) for explanations.
