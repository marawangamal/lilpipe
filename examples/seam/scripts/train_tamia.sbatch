#!/bin/bash
# Upstream SEAM commit: fa7224f8754585786c850f327b08f32a6976420e
#SBATCH --job-name=seam-defense
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00

set -euo pipefail

: "${SEAM_ROOT:?Set SEAM_ROOT to the upstream SEAM checkout}"
: "${SEAM_ENV:?Set SEAM_ENV to the prepared Python environment}"
: "${SEAM_RUN_ROOT:?Set SEAM_RUN_ROOT to the scratch run directory}"

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR="$SEAM_RUN_ROOT/wandb"

mkdir -p "$WANDB_DIR"
cd "$SEAM_ROOT"

TRAIN_ARGS=(
    --output_dir "$SEAM_RUN_ROOT/defense"
    --wandb_run_name "seam_${SEAM_DEFENSE_SIZE:-8000}_lr2e-5_eps0.001_beta0.01_alpha1"
    --learning_rate 2e-5
    --defense_size "${SEAM_DEFENSE_SIZE:-8000}"
    --epsilon 0.001
    --save_total_limit 1
)
if [[ -n "${SEAM_MAX_STEPS:-}" ]]; then
    TRAIN_ARGS+=(--max_steps "$SEAM_MAX_STEPS" --save_steps "${SEAM_SAVE_STEPS:-1}")
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SEAM_ENV/bin/python" -m src.train \
    "${TRAIN_ARGS[@]}"
