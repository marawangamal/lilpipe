#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/.cache/huggingface/datasets"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"
export PYTHONPATH="$PWD"

model_root=${1:?missing model root}
result_root=${2:?missing result root}
adapter_name_or_path=${3:?missing adapter name or path}
base_model_name_or_path=${4:?missing base model name or path}
checkpoint_frequency=${5:?missing checkpoint frequency}
step=$((${SLURM_ARRAY_TASK_ID:?this script must run as an array task} * checkpoint_frequency))
last_step=${6:-}
if [[ -n "$last_step" ]] && (( step > last_step )); then
  step=$last_step
fi

adapter_name_or_path_ckpt="$model_root/$adapter_name_or_path/checkpoint-$step"

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

output="$result_root/$adapter_name_or_path/checkpoint-$step/mmlu-no-bio"
mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$base_model_name_or_path,peft=$adapter_name_or_path_ckpt,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks mmlu_no_bio \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
