#!/bin/bash
#SBATCH --job-name=cb-ret10-trajectory
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/cb-ret10-trajectory-%j.out
set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the examples/unlearn directory}
cd "$repo_root"
[[ -f scripts/run_unlearn.sh ]] || {
    echo "ERROR: submit from examples/unlearn" >&2
    exit 1
}

tamper_root=$(cd ../tamper-resistance && pwd)
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-unlearn"
export PYTHONPATH="$repo_root"
export WANDB_DIR="$repo_root/artifacts/logs"
export WANDB_PROJECT=lp-tamper-resistance
export WANDB_NAME=cb-ret10-rm23-orth5-r8-trajectory

uv sync --project "$tamper_root" --frozen --group unlearn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

bash scripts/run_unlearn.sh \
  -a cb \
  --orth 5 \
  --rm 23 \
  --ret 10 \
  --rank 8 \
  --lr 1e-3 \
  --examples 1024 \
  --pdbs 2 \
  --dtype bf16 \
  --tag-suffix pdbs2_trajectory \
  --extra "--layers 5 10 15 20 25 30 --save_steps 5 --save_total_limit 100" \
  --execute
