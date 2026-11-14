#!/bin/bash
#SBATCH --job-name=eval-header-ablation
#SBATCH --output=artifacts/tamia/logs/eval-header-ablation-%j.out
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

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
model_name_or_path=${1:?missing model name or path}

eval_env="$UV_PROJECT_ENVIRONMENT-eval"
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen --group eval

results_dir="./artifacts/tamia/results-header-ablation/${model_name_or_path}"

"$eval_env/bin/accelerate" launch \
  --multi_gpu \
  --num_processes 4 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$model_name_or_path,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks mmlu_no_bio_biology_header,wmdp_bio_robust_no_domain_header \
  --batch_size 32 \
  --output_path "$results_dir"
