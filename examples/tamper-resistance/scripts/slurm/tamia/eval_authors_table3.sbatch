#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"

authors_root=${1:?missing authors WMDP root}
model_slug=${2:?missing model slug}
result_slug=${3:?missing result slug}
task=${4:?missing lm-eval task}
output_name=${5:?missing output name}
array_index=${SLURM_ARRAY_TASK_ID:?this script must run as an array task}
samples=(0 20 40 60)
sample_count=${samples[$array_index]}

authors_root=$(realpath "$authors_root")
model_root="$authors_root/artifacts/tamia/models/$model_slug"
if (( sample_count == 0 )); then
  checkpoint="$model_root/checkpoints"
else
  checkpoint="$model_root/examples-$sample_count/checkpoints"
fi
if [[ ! -f "$checkpoint/config.json" ]]; then
  echo "Missing model checkpoint: $checkpoint" >&2
  exit 1
fi

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

output="artifacts/tamia/evals/$result_slug/samples-$sample_count/$output_name"
mkdir -p "$output"
accelerate launch --num_processes="${SLURM_GPUS_ON_NODE:?}" -m lm_eval run \
  --model hf \
  --model_args "pretrained=$checkpoint,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks "$task" \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
