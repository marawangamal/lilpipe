#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"
export PYTHONPATH="$PWD"

artifacts_dir=${1:?missing artifacts directory}
adapter_name_or_path=${2:?missing adapter name or path}
base_model_name_or_path=${3:?missing base model name or path}
checkpoint_frequency=${4:?missing checkpoint frequency}
step=$((${SLURM_ARRAY_TASK_ID:?this script must run as an array task} * checkpoint_frequency))
last_step=${5:-}
if [[ -n "$last_step" ]] && (( step > last_step )); then
  step=$last_step
fi

# adapter path is relative to artifacts dir
adapter_name_or_path_ckpt="$artifacts_dir/models/$adapter_name_or_path/checkpoint-$step"

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

output="$artifacts_dir/evals/$adapter_name_or_path/checkpoint-$step"
mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$base_model_name_or_path,peft=$adapter_name_or_path_ckpt,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
