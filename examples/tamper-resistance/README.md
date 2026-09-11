# Unfiltered WMDP-Bio LoRA tampering trajectory

Fine-tune `EleutherAI/deep-ignorance-unfiltered` on WMDP-Bio and evaluate
robust MCQA accuracy across checkpoints. Run every command below from this
directory (`examples/tamper-resistance`).

## Setup

Request access to
[`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
and log in with `hf auth login`.

```bash
# add env vars
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export HF_HOME="$SCRATCH/.cache/huggingface"

# keep generated data, checkpoints, and results on scratch
mkdir -p "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts"
ln -sfnT "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts" artifacts

# install environments
UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train" \
uv sync --group train
UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval" \
uv sync --group eval
```

## Run

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/unfiltered-wmdp-bio-lora.yml
```

After the training and evaluation jobs finish:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval/bin/activate"
python scripts/analysis/plot_trajectory.py \
  artifacts/evals/unfiltered-wmdp-bio-lora \
  configs/results/unfiltered-wmdp-bio-lora.yml \
  artifacts/results/unfiltered-wmdp-bio-lora
```
