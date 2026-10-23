#!/bin/bash
# Reproduce the SEAM defense configuration reported in Appendix A.
#SBATCH --job-name=seam-paper-defense
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

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SEAM_ENV/bin/python" -m src.train \
    --output_dir "$SEAM_RUN_ROOT/defense" \
    --wandb_run_name "seam_paper_steps500_lr2e-5_eps0.001_beta0.01_alpha1" \
    --max_steps 500 \
    --save_steps 500 \
    --save_total_limit 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0 \
    --defense_size 8000 \
    --alpha 1 \
    --beta 0.01 \
    --epsilon 0.001
