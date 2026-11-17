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

attack_config=${1:?missing attack config}
defence_model_name_or_path=${2:?missing model name or path} # e.g., OPTML-Group/GradDiff-WMDP

train_env="$UV_PROJECT_ENVIRONMENT-train"
eval_env="$UV_PROJECT_ENVIRONMENT-eval"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --frozen --group train
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen --group eval

attack_learning_rates=(1e-5 5e-5 1e-4)
attack_epochs=(1 2 3)

# Evaluate the defended model before relearning if results are missing.
defence_results_dir="./artifacts/tamia/results/${defence_model_name_or_path}"
if [[ -z "$(find "$defence_results_dir" -type f \
  -name 'results_*.json' -print -quit 2>/dev/null)" ]]; then
  rm -rf -- "$defence_results_dir"
  "$eval_env/bin/accelerate" launch \
    --multi_gpu \
    --num_processes 4 -m lm_eval run \
    --model hf \
    --model_args "pretrained=$defence_model_name_or_path,dtype=bfloat16,trust_remote_code=True" \
    --include_path lm_eval_tasks \
    --tasks mmlu_no_bio,wmdp_bio_robust \
    --batch_size 32 \
    --output_path "$defence_results_dir"
fi

# Attack.
for attack_lr in "${attack_learning_rates[@]}"; do
  for epochs in "${attack_epochs[@]}"; do
    attack_model_name="${defence_model_name_or_path}--relearn-lr${attack_lr}-epochs${epochs}"
    attack_model_dir="./artifacts/tamia/models/${attack_model_name}"
    attack_results_dir="./artifacts/tamia/results/${attack_model_name}"
    if [[ -n "$(find "$attack_results_dir" -type f \
      -name 'results_*.json' -print -quit 2>/dev/null)" ]]; then
      echo "Skipping completed attack: $attack_model_name"
      continue
    fi

    # Restart only this incomplete attack.
    rm -rf -- "$attack_model_dir" "$attack_results_dir"
    "$train_env/bin/python" -m axolotl.cli.train "$attack_config" \
      --base_model "$defence_model_name_or_path" \
      --learning_rate "$attack_lr" \
      --num_epochs "$epochs" \
      --output_dir "$attack_model_dir" \
      --wandb_name "$attack_model_name"

    # Evaluation.
    "$eval_env/bin/accelerate" launch \
      --multi_gpu \
      --num_processes 4 -m lm_eval run \
      --model hf \
      --model_args "pretrained=$attack_model_dir,dtype=bfloat16,trust_remote_code=True" \
      --include_path lm_eval_tasks \
      --tasks mmlu_no_bio,wmdp_bio_robust \
      --batch_size 32 \
      --output_path "$attack_results_dir"

    rm -rf -- "$attack_model_dir"
  done
done
