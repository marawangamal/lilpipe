#!/bin/bash
#SBATCH --job-name=gen-ll-tar
#SBATCH --output=artifacts/tamia/logs/gen-ll-%j.out
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
# (needed for mila) tmp envs in order to avoid scratch filesystem read issues
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID" 

# (needed for tamia) compute nodes don't have access to the internet 
# except for some servers
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy

# (needed for tamia) allows access wandb server
module load httpproxy/1.0 

# ============================================
# APPLICATION SPECIFIC CONFIG 
# ============================================

export CUDA_VISIBLE_DEVICES=0

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

model=artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16
attack=$model--relearn-fft-lr5e-6-ms128
trajectory=("$model" "$attack"/checkpoint-{16,32,48,64,80,96,112,128})

python scripts/analysis/gen_ll.py \
  --model "$model" \
  --trajectory "${trajectory[@]}" \
  --dataset cais/wmdp-bio-forget-corpus \
  --radii 1e-1 1e-2 1e-3 \
  --n-pts 9 \
  --batch-size 2 \
  --seed 42 \
  --outdir artifacts/tamia/ll/tar-k16-relearn-ms128 \
  "$@"
