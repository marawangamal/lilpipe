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
released_model=${2:?missing released model}
released_revision=${3:?missing released model revision}
trained_model=${4:?missing trained model path}
result_root=${5:?missing result root}
task=${6:?missing task}
output_name=${7:?missing output name}
checkpoint_frequency=${8:-1}
array_index=${SLURM_ARRAY_TASK_ID:?this script must run as an array task}
step=$((array_index * checkpoint_frequency))

if (( step == 0 )); then
  cache_name=${released_model//\//--}
  checkpoint="$HF_HOME/hub/models--$cache_name/snapshots/$released_revision"
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "Missing released model snapshot: $checkpoint" >&2
    exit 1
  fi
else
  checkpoint="$trained_model/checkpoint-$step"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
fi

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

output="$result_root/$model_id/checkpoint-$step/$output_name"
mkdir -p "$output"
accelerate launch --num_processes="${SLURM_GPUS_ON_NODE:?}" -m lm_eval run \
  --model hf \
  --model_args "pretrained=$checkpoint,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks "$task" \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
