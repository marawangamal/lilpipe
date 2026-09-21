#!/bin/bash
#SBATCH --job-name=unlearn-cb-orth5-rm23-ret0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/cb-orth5-rm23-ret0-%j.out
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tamper_root=$(cd "$repo_root/../tamper-resistance" && pwd)
cd "$repo_root"

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-unlearn"
export PYTHONPATH="$repo_root"
export WANDB_DIR="$repo_root/artifacts/logs"
export WANDB_PROJECT=lp-tamper-resistance
export WANDB_NAME=cb-lora-ret0-rm23-orth5-r8-pdbs2-lr1e-3

uv sync --project "$tamper_root" --frozen --group unlearn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

bash scripts/run_unlearn.sh \
  -a cb \
  --orth 5 \
  --rm 23 \
  --ret 0 \
  --rank 8 \
  --lr 1e-3 \
  --examples 1024 \
  --pdbs 2 \
  --dtype bf16 \
  --tag-suffix pdbs2 \
  --extra "--layers 5 10 15 20 25 30" \
  --execute
