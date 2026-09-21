#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

module load cuda/12.1.1

model_id=${1:?missing model id}
model_name_or_path=${2:?missing model name or path}
domain=${3:?missing WMDP domain}
output_dir="artifacts/evals/$model_id"

test -x .venv/bin/python || {
  echo "Missing .venv; follow the setup instructions in README-lilpipe.md" >&2
  exit 1
}

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SLURM_TMPDIR/.cache/huggingface/datasets}"
export PYTHONPATH="$PWD"

mkdir -p "$output_dir" artifacts/logs
.venv/bin/python src/eval.py --config-name=eval.yaml \
  experiment=eval/wmdp/default \
  data_split="$domain" \
  task_name="$model_id" \
  model.model_args.pretrained_model_name_or_path="$model_name_or_path" \
  model.model_args.attn_implementation=eager \
  model.tokenizer_args.pretrained_model_name_or_path="$model_name_or_path" \
  eval.lm_eval.simple_evaluate_args.batch_size=4 \
  paths.output_dir="$output_dir"
