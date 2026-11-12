#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=artifacts/tamia/logs/train-%j.out
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

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$PWD/artifacts/tamia/logs"

defence_config=${1:?missing defence config}
attack_config=${2:?missing attack config}
model_name=${3:?missing model name} # e.g., z7b

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"



defence_learning_rates=(1e-5 5e-5 1e-6)
attack_learning_rates=(1e-5 5e-5 1e-6)
defence_retain_coefficients=(0.1 1.0 10.0)

for defence_lr in "${defence_learning_rates[@]}"; do
  for defence_retain_coefficient in "${defence_retain_coefficients[@]}"; do
    sweep_name="--unlearn-lr${defence_lr}-ret${defence_retain_coefficient}"
    ckpt_dir="./artifacts/tamia/models/${model_name}/sweeps/$sweep_name"
    results_dir="./artifacts/tamia/results/${model_name}/sweeps/$model_name"

    # Defence.
    axolotl train "$defence_config" \
      --launcher python \
      --learning-rate "$defence_lr" \
      --custom_trainer_args.retain_coefficient "$defence_retain_coefficient" \
      --output-dir "$ckpt_dir" \
      --wandb-name "$model_name"

    # Attack.
    for attack_lr in attack_learning_rates
        axolotl train "$attack_config" --base_model "$ckpt_dir" --output-dir "$ckpt_dir-${--relearn}"

    # Evaluation.
    accelerate launch -m lm_eval \
        --multi_gpu \
        --num_processes 4 \
        --model hf \
        --model_args "$ckpt_dir" \
        --include_path lm_eval_tasks \
        --tasks mmlu_no_bio,wmdp_bio_robust \
        --batch_size 32

    # Delete model artifacts only after evaluation succeeds
    rm -rf -- "$ckpt_dir"
  done
done