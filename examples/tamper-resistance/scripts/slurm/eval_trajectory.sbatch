#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%A_%a.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
mkdir -p artifacts/logs

base_model=${1:?missing base model}
adapter_dir=${2:?missing adapter directory}
_training_config=${3:?missing training config}
_milestones_config=${4:?missing milestones config}
eval_dir=${5:?missing evaluation directory}

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"
export PYTHONPATH="$PWD"

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

task_id=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task}
if (( task_id < 0 || task_id > 8 )); then
  echo "Array task $task_id is outside the configured range 0-8" >&2
  exit 1
fi
step=$((task_id * 250))
output="$eval_dir/checkpoint-$step"
mkdir -p "$output"

if [[ "$step" == 0 ]]; then
  model_args="pretrained=$base_model,dtype=bfloat16,trust_remote_code=True"
else
  checkpoint="$adapter_dir/checkpoint-$step"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Required LoRA checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
  model_args="pretrained=$base_model,peft=$checkpoint,dtype=bfloat16,trust_remote_code=True"
fi

accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "$model_args" \
  --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
