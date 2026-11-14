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
model_name_or_path=${2:?missing model name or path} # e.g., OPTML-Group/GradDiff-WMDP

train_env="$UV_PROJECT_ENVIRONMENT-train"
eval_env="$UV_PROJECT_ENVIRONMENT-eval"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --frozen --group train
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen --group eval

attack_learning_rates=(1e-5)
attack_epochs=(1 2 3)

ckpt_dir="./artifacts/tamia/models/${model_name_or_path}"
results_dir="./artifacts/tamia/results/${model_name_or_path}"

# ================================================================================
# Resumption handling. Skip iff every attack config for this defence exists
skip_defence=true
for attack_lr in "${attack_learning_rates[@]}"; do
  for epochs in "${attack_epochs[@]}"; do
    attack_results_dir="${results_dir}--relearn-lr${attack_lr}-epochs${epochs}"
    if [[ -z "$(find "$attack_results_dir" -type f \
      -name 'results_*.json' -print -quit 2>/dev/null)" ]]; then
      skip_defence=false
    fi
  done
done

if "$skip_defence"; then
  echo "Skipping completed defence: $model_name_or_path"
  exit 0
fi

# Restart this defence and all its attacks from scratch.
rm -rf -- \
  "$ckpt_dir" "$results_dir" \
  "${ckpt_dir}"--relearn-* \
  "${results_dir}"--relearn-*
# ================================================================================

# Evaluate the defended model before relearning.
"$eval_env/bin/accelerate" launch \
  --multi_gpu \
  --num_processes 4 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$model_name_or_path,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks mmlu_no_bio,wmdp_bio_robust \
  --batch_size 32 \
  --output_path "$results_dir"

# Attack.
for attack_lr in "${attack_learning_rates[@]}"; do
  for epochs in "${attack_epochs[@]}"; do
    attack_name="relearn-lr${attack_lr}-epochs${epochs}"
    attack_dir="${ckpt_dir}--${attack_name}"
    "$train_env/bin/python" -m axolotl.cli.train "$attack_config" \
      --base_model "$model_name_or_path" \
      --learning_rate "$attack_lr" \
      --num_epochs "$epochs" \
      --output_dir "$attack_dir" \
      --wandb_name "$model_name_or_path--$attack_name"

    # Evaluation.
    "$eval_env/bin/accelerate" launch \
      --multi_gpu \
      --num_processes 4 -m lm_eval run \
      --model hf \
      --model_args "pretrained=$attack_dir,dtype=bfloat16,trust_remote_code=True" \
      --include_path lm_eval_tasks \
      --tasks mmlu_no_bio,wmdp_bio_robust \
      --batch_size 32 \
      --output_path "${results_dir}--${attack_name}"

    rm -rf -- "$attack_dir"
  done
done

# Delete model artifacts only after evaluation succeeds
rm -rf -- "$ckpt_dir"
