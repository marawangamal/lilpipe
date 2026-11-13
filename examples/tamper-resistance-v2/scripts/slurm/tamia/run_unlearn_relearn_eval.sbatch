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

export PYTHONPATH="$PWD/src/data:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$PWD/artifacts/tamia/logs"

defence_config=${1:?missing defence config}
attack_config=${2:?missing attack config}
model_name=${3:?missing model name} # e.g., z7b

runtime_project="$PWD/../tamper-resistance"
train_env="$UV_PROJECT_ENVIRONMENT-train"
eval_env="$UV_PROJECT_ENVIRONMENT-eval"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --project "$runtime_project" --frozen --group train
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --project "$runtime_project" --frozen --group eval


defence_learning_rates=(1e-5 5e-5 1e-6)
attack_learning_rates=(1e-5 5e-5 1e-6)
defence_retain_coefficients=(0.1 1.0 10.0)

for defence_lr in "${defence_learning_rates[@]}"; do
  for defence_retain_coefficient in "${defence_retain_coefficients[@]}"; do
    sweep_name="unlearn-lr${defence_lr}-ret${defence_retain_coefficient}"
    ckpt_dir="./artifacts/tamia/models/${model_name}/sweeps/$sweep_name"
    results_dir="./artifacts/tamia/results/${model_name}/sweeps/$sweep_name"

    # Defence.
    "$train_env/bin/axolotl" train "$defence_config" \
      --launcher python \
      --learning_rate "$defence_lr" \
      --custom_trainer_args__coeff_retain "$defence_retain_coefficient" \
      --output_dir "$ckpt_dir" \
      --wandb_name "$model_name--$sweep_name"

    # Attack.
    for attack_lr in "${attack_learning_rates[@]}"; do
      attack_dir="${ckpt_dir}--relearn-lr${attack_lr}"
      "$train_env/bin/axolotl" train "$attack_config" \
        --launcher python \
        --base_model "$ckpt_dir" \
        --learning_rate "$attack_lr" \
        --output_dir "$attack_dir" \
        --wandb_name "$model_name--$sweep_name--relearn-lr${attack_lr}"

      # Evaluation.
      "$eval_env/bin/accelerate" launch \
        --multi_gpu \
        --num_processes 4 -m lm_eval run \
        --model hf \
        --model_args "pretrained=$attack_dir,dtype=bfloat16,trust_remote_code=True" \
        --include_path "$runtime_project/lm_eval_tasks" \
        --tasks mmlu_no_bio,wmdp_bio_robust \
        --batch_size 32 \
        --output_path "$results_dir/relearn-lr${attack_lr}"

      rm -rf -- "$attack_dir"
    done

    # Delete model artifacts only after evaluation succeeds
    rm -rf -- "$ckpt_dir"
  done
done
