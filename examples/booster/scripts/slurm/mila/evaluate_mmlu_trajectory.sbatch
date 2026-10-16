#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-10
#SBATCH --output=artifacts/mila/logs/slurm-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

method=${1:?missing method}
attack=${2:-harmful-only}
step=$((SLURM_ARRAY_TASK_ID * 200))
mkdir -p artifacts/mila/logs

child=""
requeue() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
}
trap requeue USR1

uv sync --project ../tamper-resistance --offline --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python scripts/lilpipe/evaluate_mmlu_trajectory.py \
    --method "$method" \
    --attack "$attack" \
    --step "$step" &
child=$!
wait "$child"
