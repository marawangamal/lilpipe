#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

model_id=${1:?missing model id}
base_model_name_or_path=${2:?missing base model name or path}
adapter_name_or_path=${3:?missing adapter name or path}
result_root=${4:?missing result root}
checkpoint_frequency=${5:?missing checkpoint frequency}
last_step=${6:?missing last step}
step=$((${SLURM_ARRAY_TASK_ID:?this script must run as an array task} * checkpoint_frequency))
if (( step > last_step )); then
  step=$last_step
fi

model_args="pretrained=$base_model_name_or_path,dtype=bfloat16,trust_remote_code=True"
if [[ "$adapter_name_or_path" == "-" ]]; then
  checkpoint="$base_model_name_or_path/checkpoint-$step"
  model_args="pretrained=$checkpoint,dtype=bfloat16,trust_remote_code=True"
else
  checkpoint="$adapter_name_or_path/checkpoint-$step"
  model_args+=",peft=$checkpoint"
fi
if [[ ! -d "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 1
fi

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

output="$result_root/$model_id/checkpoint-$step/mmlu-no-bio"
mkdir -p "$output"
accelerate launch --num_processes="${SLURM_GPUS_ON_NODE:?}" -m lm_eval run \
  --model hf \
  --model_args "$model_args" \
  --include_path lm_eval_tasks \
  --tasks mmlu_no_bio \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
