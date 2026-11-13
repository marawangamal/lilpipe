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

defence_config=${1:?missing defence config}
attack_config=${2:?missing attack config}
model_name=${3:?missing model name} # e.g., z7b

train_env="$UV_PROJECT_ENVIRONMENT-train"
eval_env="$UV_PROJECT_ENVIRONMENT-eval"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --frozen --group train
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen --group eval


defence_learning_rates=(1e-6 5e-6)
attack_learning_rates=(1e-5)
attack_epochs=(1 2 3)
defence_retain_coefficients=(1.0 1.5 2.0)

for defence_lr in "${defence_learning_rates[@]}"; do
  for defence_retain_coefficient in "${defence_retain_coefficients[@]}"; do
    sweep_name="unlearn-lr${defence_lr}-ret${defence_retain_coefficient}"
    ckpt_dir="./artifacts/tamia/models/${model_name}/sweeps/$sweep_name"
    results_dir="./artifacts/tamia/results/${model_name}/sweeps/$sweep_name"

    # Defence.
    "$train_env/bin/python" -m axolotl.cli.train "$defence_config" \
      --learning_rate "$defence_lr" \
      --custom_trainer_args__coeff_retain "$defence_retain_coefficient" \
      --output_dir "$ckpt_dir" \
      --wandb_name "$model_name--$sweep_name"

    # Attack.
    for attack_lr in "${attack_learning_rates[@]}"; do
      for epochs in "${attack_epochs[@]}"; do
        attack_name="relearn-lr${attack_lr}-epochs${epochs}"
        attack_dir="${ckpt_dir}--${attack_name}"
        "$train_env/bin/python" -m axolotl.cli.train "$attack_config" \
          --base_model "$ckpt_dir" \
          --learning_rate "$attack_lr" \
          --num_epochs "$epochs" \
          --output_dir "$attack_dir" \
          --wandb_name "$model_name--$sweep_name--$attack_name"

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
  done
done
