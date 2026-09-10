#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%A_%a.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
mkdir -p artifacts/logs

base_model=${1:?missing base model}
adapter_dir=${2:?missing adapter directory}
training_config=${3:?missing training config}
milestones_config=${4:?missing milestones config}
eval_dir=${5:?missing evaluation directory}

source .venv-train/bin/activate
axolotl_bin=$VIRTUAL_ENV/bin/axolotl
mapfile -t milestones < <(python - "$milestones_config" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    for milestone in yaml.safe_load(handle)["milestones"]:
        print(milestone)
PY
)
task_id=${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array task}
if (( task_id < 0 || task_id >= ${#milestones[@]} )); then
  echo "Array task $task_id has no configured milestone" >&2
  exit 1
fi
step=${milestones[$task_id]}

evaluate() {
  local model=$1
  local step=$2
  local output="$eval_dir/checkpoint-$step"
  mkdir -p "$output"
  source .venv-eval/bin/activate
  accelerate launch --num_processes=1 -m lm_eval run \
    --model hf \
    --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True" \
    --include_path lm_eval_tasks \
    --tasks wmdp_bio_robust \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$output"
  source .venv-train/bin/activate
}

if [[ "$step" == 0 ]]; then
  evaluate "$base_model" "$step"
  exit 0
fi

checkpoint="$adapter_dir/checkpoint-$step"
if [[ ! -d "$checkpoint" ]]; then
  echo "Required LoRA checkpoint is missing: $checkpoint" >&2
  exit 1
fi
merge_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lilpipe-merge-${step}-XXXXXX")
trap 'rm -rf -- "$merge_dir"' EXIT
"$axolotl_bin" merge-lora "$training_config" \
  --base-model "$base_model" \
  --lora-model-dir "$checkpoint" \
  --output-dir "$merge_dir"
evaluate "$merge_dir/merged" "$step"
