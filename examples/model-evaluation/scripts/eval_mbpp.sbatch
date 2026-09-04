#!/bin/bash
set -euo pipefail

base_model=$1
adapter=$2
merge_config=$3
output=$4
model=$base_model

if [[ "$adapter" != none ]]; then
  merge_dir="${SLURM_TMPDIR:-/tmp}/lilpipe-${SLURM_JOB_ID:-local}"
  axolotl merge-lora "$merge_config" \
    --base-model "$base_model" \
    --lora-model-dir "$adapter" \
    --output-dir "$merge_dir"
  model="$merge_dir/merged"
fi

mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$model,dtype=float16,trust_remote_code=True,enable_thinking=False" \
  --tasks mbpp_evalplus \
  --include_path lm_eval_tasks \
  --batch_size auto \
  --apply_chat_template \
  --gen_kwargs max_gen_toks=1024,do_sample=False \
  --output_path "$output/results.json" \
  --log_samples \
  --confirm_run_unsafe_code

python scripts/report_mbpp.py "$output"
