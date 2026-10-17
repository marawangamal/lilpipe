#!/bin/bash
#SBATCH --job-name=smoothness
#SBATCH --output=artifacts/tamia/logs/smoothness-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export PATH="$HOME/.local/bin:$PATH"

gd=artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-gd/checkpoint-125
sam=artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2/checkpoint-125
for checkpoint in "$gd" "$sam"; do
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "Missing model checkpoint: $checkpoint" >&2
    exit 1
  fi
done

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python scripts/analysis/plot_smoothness.py \
  --model Origin HuggingFaceH4/zephyr-7b-beta \
  --model GradDiff "$gd" \
  --model "GradDiff + SAM" "$sam" \
  "$@"
