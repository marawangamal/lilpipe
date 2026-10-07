# Tamper-resistance experiments

The active experiment family is the matched Zephyr-7B LoRA `s125` study: 125
unlearning steps followed by 125 relearning steps for GD and GD+SAM at
`rho={1e-3, 1e-2, 1e-1}`. Run all commands from
`examples/tamper-resistance`, because lilpipe resolves paths from the current
working directory.

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

Run the GD and GD+SAM unlearning learning-rate sweeps separately:

```bash
lilpipe configs/experiments/z7b-lora-s125-unlearn-gd-hpo.yml
lilpipe configs/experiments/z7b-lora-s125-unlearn-gd-sam-hpo.yml
```

The Tamia full-finetuning GD study uses the paper's WMDP learning-rate range as
the same four-point grid for unlearning and relearning:
`2.5e-6, 5e-6, 7.5e-6, 1e-5`.

```bash
lilpipe configs/experiments/z7b-fft-s125-unlearn-gd-hpo.yml
# Promote the selected unlearned model before running:
lilpipe configs/experiments/z7b-fft-s125-relearn-gd-hpo.yml
```

The matching full-finetuning GD+SAM unlearning sweep crosses the same learning
rates with `rho={1e-3, 1e-2, 1e-1}`:

```bash
lilpipe configs/experiments/z7b-fft-s125-unlearn-gd-sam-hpo.yml
# After promoting the selected rho and learning rate:
lilpipe configs/experiments/z7b-fft-s125-relearn-gd-sam-rho1e-3-hpo.yml
```

After promoting the selected unlearning models, run the corresponding
relearning learning-rate sweeps:

```bash
lilpipe configs/experiments/z7b-lora-s125-relearn-gd-hpo.yml
lilpipe configs/experiments/z7b-lora-s125-relearn-gd-sam-hpo.yml
```

Run the selected canonical unlearning and relearning configurations with:

```bash
lilpipe configs/experiments/z7b-lora-s125.yml
```

LoRA runs save checkpoints every 10 steps plus step 125. Full-finetuning
sweeps retain only step 125 to avoid redundant full-model checkpoints. Sweep
manifests evaluate the final checkpoint; the canonical manifest evaluates the
full trajectory.

Retired experiment artifacts are kept under
`artifacts/archive/non-s125-20261005/`.

Axolotl loads dataset strategies from `configs/training/data/` and objectives
from `configs/training/trainers/`, as selected by `datasets[].type` and
`trainer_cls` in each YAML.
