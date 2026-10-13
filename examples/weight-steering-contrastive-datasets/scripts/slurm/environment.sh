#!/bin/bash
set -euo pipefail
module load cuda/12.1.1
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface/datasets"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
environment_archive="$SCRATCH/lilpipe/examples/weight-steering-contrastive-datasets/open-unlearning-venv.tar"
environment_root="$SLURM_TMPDIR/open-unlearning-environment"
test -f "$environment_archive" || {
    echo "Missing environment archive: $environment_archive" >&2
    exit 1
}
mkdir -p "$environment_root"
tar -C "$environment_root" -xf "$environment_archive"
export EXPERIMENT_PYTHON="$environment_root/.venv/bin/python"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PATH="$HOME/.local/bin:$PATH"
test -x "$EXPERIMENT_PYTHON" || {
    echo "Missing examples/open-unlearning environment" >&2
    exit 1
}
mkdir -p artifacts/logs
