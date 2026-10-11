# Tamper-resistance experiments

The active Zephyr-7B experiments are organized as self-contained unlearn,
resist, and relearn lineages under `configs/stages/`. Run every command from
this directory because lilpipe resolves paths from the current working
directory.

## Setup

Request access to
[`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
and log in with `hf auth login`.

```bash
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export HF_HOME="$SCRATCH/.cache/huggingface"

mkdir -p "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts"
ln -sfnT "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts" artifacts

UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train" \
uv sync --group train
UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval" \
uv sync --group eval
```

The example installs its `trainers`, `data`, and `axolotl_plugins` packages
from `src/`. Custom trainer settings live in YAML under
`custom_trainer_args`; the plugin validates and forwards them to the trainer
constructor.

## Generate and run a sweep

Each lineage has a runnable `base.yml`, a parameter `sweep.yml`, and a
promoted standalone `selected.yml`. Generate candidates from a sweep with:

```bash
python scripts/hpo.py generate \
  configs/stages/0_unlearn/z7b-wmdp-bio--unlearn-fft-gd/sweep.yml
lilpipe artifacts/generated/hpo/z7b-wmdp-bio--unlearn-fft-gd-hpo/experiment.yml
```

Generated configurations, their model registry, and their manifest are kept
under `artifacts/generated/hpo/<sweep-id>/`. HPO runs keep only the final
checkpoint. Selection is manual: copy the chosen concrete values to the
lineage's `selected.yml` and promote its model in
`configs/registries/models.yml`.

## Run canonical lineages

```bash
lilpipe configs/experiments/unlearn.yml
lilpipe configs/experiments/unlearn-relearn.yml
lilpipe configs/experiments/unlearn-resist-relearn.yml
```

Canonical relearning configurations retain trajectory checkpoints for
evaluation. Existing artifact directories are not renamed or removed.
