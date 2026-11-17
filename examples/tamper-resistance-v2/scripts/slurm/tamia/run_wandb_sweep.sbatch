#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=artifacts/tamia/logs/sweep-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$PWD/artifacts/tamia/cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

# Tamia compute nodes use cached packages, models, and datasets.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export AXOLOTL_DO_NOT_TRACK=1

# Allow access to the W&B server.
module load httpproxy/1.0

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

export PYTHONPATH="$PWD/src/data:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$PWD/artifacts/tamia"
export WANDB_DATA_DIR="$PWD/artifacts/tamia/cache/wandb/data"
export WANDB_CACHE_DIR="$PWD/artifacts/tamia/cache/wandb"
mkdir -p "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"

sweep=${1:?missing sweep YAML or entity/project/sweep ID}
trial_count=${2:-10}

train_env="$UV_PROJECT_ENVIRONMENT-train"
eval_env="$UV_PROJECT_ENVIRONMENT-eval"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --frozen --group train
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen --group eval
export PATH="$train_env/bin:$PATH"
export EVAL_ACCELERATE="$eval_env/bin/accelerate"

if [[ -f "$sweep" ]]; then
  sweep_id=$("$train_env/bin/python" scripts/misc/create_wandb_sweep_id.py "$sweep")
else
  sweep_id="$sweep"
fi

echo "Sweep ID: $sweep_id"
echo "Continue with: sbatch --partition=gpubase_bynode_b1 scripts/slurm/tamia/run_wandb_sweep.sbatch $sweep_id $trial_count"
"$train_env/bin/wandb" agent --count "$trial_count" "$sweep_id"
