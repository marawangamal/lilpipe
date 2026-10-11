#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

model_id=${1:?missing model id}
trained_model=${2:?missing trained model path}
result_root=${3:?missing result root}
task=${4:?missing task}
output_name=${5:?missing output name}
shift 5
steps=("$@")
if (( ${#steps[@]} == 0 )); then
  steps=(5 10 15)
fi

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

for step in "${steps[@]}"; do
  checkpoint="$trained_model/checkpoint-$step"
  if [[ ! -f "$checkpoint/config.json" ]]; then
    echo "Missing model checkpoint: $checkpoint" >&2
    exit 1
  fi
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
done
