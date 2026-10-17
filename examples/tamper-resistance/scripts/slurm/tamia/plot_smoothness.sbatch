#!/bin/bash
#SBATCH --job-name=smoothness
#SBATCH --output=artifacts/tamia/logs/smoothness-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=01:00:00
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

# Direction scales to evaluate; every model shares the same seeded directions.
read -r -a direction_stds <<<"${DIRECTION_STDS:-0.003 0.01 0.03 0.1}"

# slug|label|model, in figure order.
models=(
  "origin|Base model|HuggingFaceH4/zephyr-7b-beta"
  "optml-npo|OPTML NPO|OPTML-Group/NPO-WMDP"
  "optml-npo-sam|OPTML NPO + SAM|OPTML-Group/NPO-SAM-WMDP"
  "npo-5pp|NPO (5pp)|artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-npo/checkpoint-125"
  "npo-sam-5pp|NPO + SAM (5pp)|artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2/checkpoint-125"
  "npo-15pp|NPO (15pp)|artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-npo/checkpoint-125"
  "npo-sam-15pp|NPO + SAM (15pp)|artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2/checkpoint-125"
  # GradDiff is only trained under the 15pp budget.
  "grad-diff-15pp|GradDiff (15pp)|artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-gd/checkpoint-125"
  "grad-diff-sam-15pp|GradDiff + SAM (15pp)|artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2/checkpoint-125"
  # Vanilla MaxEnt has no 15pp model yet.
  "me-5pp|MaxEnt (5pp)|artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me/checkpoint-125"
  "me-sam-5pp|MaxEnt + SAM (5pp)|artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2/checkpoint-125"
  "me-sam-15pp|MaxEnt + SAM (15pp)|artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-me-sam-rho1e-2/checkpoint-125"
)
for entry in "${models[@]}"; do
  IFS='|' read -r _ _ model <<<"$entry"
  if [[ $model == artifacts/* && ! -f "$model/config.json" ]]; then
    echo "Missing model checkpoint: $model" >&2
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
    --direction-std "${direction_stds[@]}" \
    --output "$part_dir/$slug.json" \
    "$@" >"$part_dir/$slug.log" 2>&1
}

# Each GPU works through every fourth model in turn.
run_lane() {
  local gpu=$1
  shift
  local index slug label model
  for ((index = gpu; index < ${#models[@]}; index += 4)); do
    IFS='|' read -r slug label model <<<"${models[index]}"
    run_surface "$gpu" "$slug" "$label" "$model" "$@"
  done
}

pids=()
for gpu in 0 1 2 3; do
  run_lane "$gpu" "$@" &
  pids+=($!)
done

echo "Per-model logs: $part_dir/*.log"
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for pid in "${pids[@]}"; do
  wait "$pid"
done
trap - EXIT

inputs=()
for entry in "${models[@]}"; do
  IFS='|' read -r slug _ _ <<<"$entry"
  inputs+=("$part_dir/$slug.json")
done
for direction_std in "${direction_stds[@]}"; do
  tag=$(python -c "print(f'{float(\"$direction_std\"):.0e}'.replace('e-0', 'e-'))")
  python scripts/analysis/plot_smoothness.py "${inputs[@]}" \
    --direction-std "$direction_std" \
    --output "artifacts/figures/z7b-wmdp-bio-loss-landscape-sigma$tag-bf16.pdf"
done
