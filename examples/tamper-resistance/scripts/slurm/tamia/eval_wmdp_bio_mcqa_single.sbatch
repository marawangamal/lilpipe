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
base_model_name_or_path=${2:?missing base model name or path}
adapter_name_or_path=${3:?missing adapter name or path}
result_root=${4:?missing result root}
output="$result_root/$model_id/wmdp-bio-robust"
model_args="pretrained=$base_model_name_or_path,dtype=bfloat16,trust_remote_code=True"
if [[ "$adapter_name_or_path" != "-" ]]; then
  model_args+=",peft=$adapter_name_or_path"
fi

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
mkdir -p "$output"
accelerate launch --num_processes="${SLURM_GPUS_ON_NODE:?}" -m lm_eval run \
  --model hf \
  --model_args "$model_args" \
  --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
