#!/bin/bash
#SBATCH --job-name=model-dist
#SBATCH --output=artifacts/tamia/logs/model-dist-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=02:00:00
# Usage: sbatch scripts/slurm/tamia/model_dist.sbatch OUT REF MODEL [MODEL ...]
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

out=${1:?missing output json}
ref=${2:?missing reference model}
shift 2
if (($# == 0)); then
  echo "missing models to compare" >&2
  exit 1
fi

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python scripts/analysis/model_dist.py "$ref" "$@" --out "$out"
