#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

method=${1:?missing unlearning method}
domain=${2:?missing WMDP domain}
output_dir=${3:?missing output directory}

test -x .venv/bin/python || {
  echo "Missing .venv; follow the setup instructions in README-lilpipe.md" >&2
  exit 1
}
test -f "data/wmdp/wmdp-corpora/${domain}-forget-corpus.jsonl" || {
  echo "Missing WMDP ${domain} corpus; run: .venv/bin/python setup_data.py --wmdp" >&2
  exit 1
}

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SLURM_TMPDIR/.cache/huggingface/datasets}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$output_dir" artifacts/logs
.venv/bin/python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/wmdp/default \
  trainer="$method" \
  data_split="$domain" \
  task_name="$(basename "$output_dir")" \
  paths.output_dir="$output_dir" \
  model.model_args.attn_implementation=eager \
  trainer.args.do_eval=false \
  trainer.args.eval_strategy=no \
  trainer.args.report_to=none
