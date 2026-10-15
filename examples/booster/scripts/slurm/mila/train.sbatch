#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

python_bin="$SCRATCH/lilpipe/envs/booster-figure3/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing prepared environment: $python_bin" >&2
    exit 2
fi

mkdir -p artifacts/mila/logs

child=""
requeue() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap requeue USR1

"$python_bin" scripts/lilpipe/train.py "$@" &
child=$!
wait "$child"
