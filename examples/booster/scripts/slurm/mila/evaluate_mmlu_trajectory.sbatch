#!/bin/bash
#SBATCH --partition=long
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
export PATH="$HOME/.local/bin:$PATH"

python_bin="../tamper-resistance/.venv-eval/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing prepared evaluation environment: $python_bin" >&2
    exit 2
fi

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

"$python_bin" scripts/lilpipe/evaluate_mmlu_trajectory.py \
    --method "$method" \
    --attack "$attack" \
    --step "$step" &
child=$!
wait "$child"
