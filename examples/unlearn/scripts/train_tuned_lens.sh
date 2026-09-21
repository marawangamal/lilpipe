#!/bin/bash
# Train tuned lens for EleutherAI/deep-ignorance-unfiltered

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export CUDA_VISIBLE_DEVICES=1

cd "$REPO_ROOT"

python -m unlearn.tuned_lens.train \
    --model_name "EleutherAI/deep-ignorance-unfiltered" \
    --bio_forget_path "data/bio_forget_ds" \
    --output_dir "runs/tuned_lens" \
    --num_epochs 3 \
    --batch_size 8 \
    --lr 1e-3 \
    --wandb_project "tuned-lens" \
    --wandb_run_name "deep-ignorance-unfiltered-lens"
