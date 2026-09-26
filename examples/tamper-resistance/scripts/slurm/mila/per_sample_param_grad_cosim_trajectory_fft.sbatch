#!/bin/bash
#SBATCH --output=artifacts/mila/logs/per-sample-grad-cosim-%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=cn-c034,cn-l030,cn-l055
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

output_dir="artifacts/mila/analysis/per_sample_param_grad_cosim_fft/$model_id"
mkdir -p "$output_dir"

mapfile -t checkpoints < <(
  find "$model_dir" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    -printf '%f\n' | sort -t- -k2,2n
)

for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint="$model_dir/$checkpoint_name"
  if [[ ! -f "$checkpoint/model.safetensors" ]]; then
    continue
  fi

  out_dir="$output_dir/$checkpoint_name"
  python scripts/analysis/per_sample_param_grad_cosim_gen_results_fft.py \
    --model_name_or_path "$checkpoint" \
    --dataset cais/wmdp-bio-forget-corpus \
    --max_samples 16 \
    --out "$out_dir"

  # reclaim the per-sample fp32 grads (~29GB each); keep only cosim.json
  rm -rf "$out_dir/grads"
done
