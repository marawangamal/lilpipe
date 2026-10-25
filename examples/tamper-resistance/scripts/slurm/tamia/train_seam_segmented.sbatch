#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
#SBATCH --requeue
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export WANDB_DIR="$PWD/artifacts/tamia/logs"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_DATA_DIR="$SCRATCH/.cache/wandb/staging"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SEAM_SEGMENT_STEPS="${SEAM_SEGMENT_STEPS:-20}"

config=${1:?missing training config}
final_output=${2:?missing final output directory}
total_steps=${3:-500}
work="$SLURM_TMPDIR/seam-segments"
checkpoint_store="${final_output}.seam-checkpoint"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$work"
module load httpproxy/1.0
uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

resume=()
if [[ -f "$checkpoint_store/complete-step" ]]; then
    step=$(<"$checkpoint_store/complete-step")
    checkpoint="$work/checkpoint-$step"
    mkdir -p "$checkpoint"
    rsync -a --exclude=complete-step "$checkpoint_store/" "$checkpoint/"
    resume=(--resume-from-checkpoint "$checkpoint")
fi

axolotl train "$config" \
    --launcher python \
    --output-dir "$work" \
    --max-steps "$total_steps" \
    --no-save-only-model \
    --save-total-limit 1 \
    --wandb-mode disabled \
    "${resume[@]}"

checkpoint=$(find "$work" -maxdepth 1 -type d -name 'checkpoint-*' \
    | sort -V | tail -n 1)
if [[ -z "$checkpoint" ]]; then
    echo "SEAM training stopped without writing a checkpoint" >&2
    exit 1
fi
step=${checkpoint##*-}
if (( step < total_steps )); then
    mkdir -p "$checkpoint_store"
    rsync -a --inplace --delete "$checkpoint/" "$checkpoint_store/"
    printf '%s\n' "$step" > "$checkpoint_store/complete-step"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

case "$checkpoint_store" in
    artifacts/tamia/models/*.seam-checkpoint) rm -rf -- "$checkpoint_store" ;;
    *) echo "refusing unexpected checkpoint path: $checkpoint_store" >&2; exit 1 ;;
esac

mkdir -p "$final_output"
rsync -a --delete --exclude='checkpoint-*' "$work/" "$final_output/"
