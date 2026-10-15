#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_INSTALLS=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

model_id=${1:?missing model id}
baseline_model=${2:?missing baseline model path}
trained_model=${3:?missing trained model path}
result_root=${4:?missing result root}
task=${5:?missing task}
output_name=${6:?missing output name}
checkpoint_frequency=${7:-1}
array_index=${SLURM_ARRAY_TASK_ID:?this script must run as an array task}
step=$((array_index * checkpoint_frequency))

if (( step == 0 )); then
  checkpoint=$baseline_model
else
  checkpoint="$trained_model/checkpoint-$step"
fi
if [[ "$checkpoint" != */* || -e "$checkpoint" ]] && [[ ! -f "$checkpoint/config.json" ]]; then
  echo "Missing model checkpoint: $checkpoint" >&2
  exit 1
fi

mkdir -p "$UV_CACHE_DIR"
install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group eval
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

output="$result_root/$model_id/checkpoint-$step/$output_name"
mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$checkpoint,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks "$task" \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path "$output"
