#!/bin/bash
#SBATCH --output=artifacts/logs/per-sample-grad-cosim-%j.out
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-train-$SLURM_JOB_ID"
export PYTHONPATH="$PWD"

model_dir=${1:?missing model directory}
model_id=$(basename "$model_dir")

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

base_model=$(python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' \
  "$model_dir/adapter_config.json")
output_dir="artifacts/analysis/per_sample_param_grad_cosim/$model_id"
mkdir -p "$output_dir"

mapfile -t checkpoints < <(
  find "$model_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    -printf '%f\n' | sort -t- -k2,2n
)

for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint="$model_dir/$checkpoint_name"
  if [[ ! -f "$checkpoint/adapter_config.json" || \
        ! -f "$checkpoint/adapter_model.safetensors" ]]; then
    continue
  fi

  python scripts/analysis/per_sample_param_grad_cosim_gen_results.py \
    --model_name_or_path "$base_model" \
    --adapter_name_or_path "$checkpoint" \
    --dataset cais/wmdp-bio-forget-corpus \
    --max_samples 16 \
    --out "$output_dir/$checkpoint_name.json"
done
