#!/bin/bash
#SBATCH --job-name=cb-ret10-relearning
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/cb-ret10-relearning-%j.out
set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the examples/unlearn directory}
cd "$repo_root"
[[ -f unlearn/__init__.py ]] || {
    echo "ERROR: submit from examples/unlearn: sbatch slurm/finetune_forget.sh" >&2
    exit 1
}

config=${1:-configs/finetune/cb-ret10-wmdp-bio-lora.yml}
tamper_root=$(cd ../tamper-resistance && pwd)

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-unlearn-finetune"
export PYTHONPATH="$repo_root"
export WANDB_DIR="$repo_root/artifacts/logs"

uv sync --project "$tamper_root" --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
axolotl train "$config" --launcher python
