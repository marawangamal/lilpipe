#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%A_%a.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
local_dir=${1:?missing local CB output directory}
eval_dir=${2:?missing evaluation directory}
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-eval"
export PYTHONPATH="$PWD"
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

case ${SLURM_ARRAY_TASK_ID:?this script must run as an array task} in
  0) label=base; model=EleutherAI/deep-ignorance-unfiltered ;;
  1) label=released; model=EleutherAI/deep-ignorance-unfiltered-cb ;;
  2|3|4)
    step=$(((SLURM_ARRAY_TASK_ID - 1) * 50))
    label=checkpoint-$step
    model=EleutherAI/deep-ignorance-unfiltered
    adapter="$local_dir/checkpoint-$step"
    [[ -d "$adapter" ]] || { echo "Missing adapter: $adapter" >&2; exit 1; }
    ;;
  *) echo "Array index must be in 0-4" >&2; exit 1 ;;
esac
model_args="pretrained=$model,dtype=bfloat16,trust_remote_code=True"
[[ -z ${adapter:-} ]] || model_args="$model_args,peft=$adapter"
output="$eval_dir/$label"
mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf --model_args "$model_args" --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust --num_fewshot 0 \
  --batch_size 32 --output_path "$output"
