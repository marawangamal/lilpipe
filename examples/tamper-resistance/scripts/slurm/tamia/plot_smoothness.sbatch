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

part_dir="artifacts/tamia/smoothness/$SLURM_JOB_ID"
mkdir -p "$part_dir"

run_surface() {
  local gpu=$1
  local slug=$2
  local label=$3
  local model=$4
  shift 4
  CUDA_VISIBLE_DEVICES=$gpu python scripts/analysis/gen_smoothness.py \
    --model "$label" "$model" \
    --dtype bfloat16 \
    --batch-size 2 \
    --direction-std 0.03 \
    --output "$part_dir/$slug.json" \
    "$@" >"$part_dir/$slug.log" 2>&1
}

run_surface 0 origin Origin HuggingFaceH4/zephyr-7b-beta "$@" &
pids=($!)
run_surface 1 grad-diff GradDiff "$gd" "$@" &
pids+=($!)
run_surface 2 grad-diff-sam "GradDiff + SAM" "$sam" "$@" &
pids+=($!)

echo "Per-model logs: $part_dir/{origin,grad-diff,grad-diff-sam}.log"
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for pid in "${pids[@]}"; do
  wait "$pid"
done
trap - EXIT

python scripts/analysis/plot_smoothness.py \
  "$part_dir/origin.json" \
  "$part_dir/grad-diff.json" \
  "$part_dir/grad-diff-sam.json" \
  --output artifacts/figures/z7b-wmdp-bio-loss-landscape--15pp-sigma3e-2-bf16.png
