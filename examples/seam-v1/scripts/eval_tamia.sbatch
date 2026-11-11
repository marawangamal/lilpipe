#!/bin/bash
# Upstream SEAM commit: fa7224f8754585786c850f327b08f32a6976420e
# Submit as an array with indices 0-4 after the defense job succeeds.
#SBATCH --job-name=seam-eval
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-4

set -euo pipefail

: "${SEAM_ROOT:?Set SEAM_ROOT to the upstream SEAM checkout}"
: "${SEAM_ENV:?Set SEAM_ENV to the prepared Python environment}"
: "${SEAM_RUN_ROOT:?Set SEAM_RUN_ROOT to the scratch run directory}"
: "${SEAM_MODEL:?Set SEAM_MODEL to the completed defense checkpoint}"

LEARNING_RATES=(2e-5 5e-5 8e-5 1e-4 2e-4)
LEARNING_RATE=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
PRE_UTILITY=False
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    PRE_UTILITY=True
fi

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR="$SEAM_RUN_ROOT/wandb"

mkdir -p "$WANDB_DIR" "$SEAM_RUN_ROOT/eval/lr-$LEARNING_RATE"
cd "$SEAM_ROOT"

# Tamia compute nodes cannot call OpenAI. Replace only the auxiliary GPT judge;
# Figure 2's local harmfulness classifier and utility evaluation are unchanged.
CUDA_VISIBLE_DEVICES=0 "$SEAM_ENV/bin/python" -c \
    'import src.eval as seam_eval; seam_eval.eval_harmfulness_gpt = lambda *args, **kwargs: float("nan"); seam_eval.main()' \
    --output_dir "$SEAM_RUN_ROOT/eval/lr-$LEARNING_RATE" \
    --model_name "$SEAM_MODEL" \
    --attack_size 1000 \
    --learning_rate "$LEARNING_RATE" \
    --pre_utility "$PRE_UTILITY" \
    --wandb_run_name "seam_eval_sample1000_lr$LEARNING_RATE"
