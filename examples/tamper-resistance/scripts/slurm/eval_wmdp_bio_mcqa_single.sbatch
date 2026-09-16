#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"
export PYTHONPATH="$PWD"

model_id=${1:?missing model id}
base_model_name_or_path=${2:?missing base model name or path}
adapter_name_or_path="artifacts/models/$model_id"
output="artifacts/evals/$model_id/wmdp-bio-robust"

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$base_model_name_or_path,peft=$adapter_name_or_path,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
