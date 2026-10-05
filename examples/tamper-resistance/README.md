# Tamper-resistance experiments

The active experiment family is the matched Zephyr-7B LoRA `s125` study: 125
unlearning steps followed by 125 relearning steps for GD and GD+SAM. Run all
commands from `examples/tamper-resistance`, because lilpipe resolves paths from
the current working directory.

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

## Run

The unlearning sweep compares GD and GD+SAM across the same learning rates.
GD+SAM uses its trainer default `rho=1e-2`; the rho suffix makes that fixed
setting explicit in artifact names.

```bash
lilpipe configs/experiments/z7b-lora-s125-hpo-unlearn-gd-sam.yml
```

After promoting the selected GD and GD+SAM unlearning models, sweep the
relearning learning rate:

```bash
lilpipe configs/experiments/z7b-lora-s125-hpo-relearn-gd-sam.yml
```

Run the selected canonical unlearning and relearning configurations with:

```bash
lilpipe configs/experiments/z7b-lora-s125.yml
```

HPO runs save only their final adapters. Canonical runs save checkpoints every
10 steps plus step 125 for trajectory evaluation.

Retired experiment artifacts are kept under
`artifacts/archive/non-s125-20261005/`.

Axolotl loads dataset strategies from `configs/training/data/` and objectives
from `configs/training/trainers/`, as selected by `datasets[].type` and
`trainer_cls` in each YAML.
