#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

module load cuda/12.1.1

source_model=${1:?missing source model}
domain=${2:?missing WMDP domain}
output_dir=${3:?missing output directory}
parameter_policy=${4:?missing parameter policy}

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/.cache/huggingface/datasets"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-open-unlearning-$SLURM_JOB_ID"
export PYTHONPATH="$PWD"
export WANDB_MODE="${WANDB_MODE:-disabled}"

uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
uv pip install --offline --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e '.[lm-eval]'
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

test -f "$source_model/config.json"
test -f "data/wmdp/wmdp-corpora/${domain}-forget-corpus.jsonl"

trainer_config=SelectiveFinetune
if [[ "$parameter_policy" == "rmu" ]]; then
  trainer_config=SelectiveFinetuneRMU
fi

mkdir -p "$output_dir" artifacts/logs
python src/train.py --config-name=train.yaml \
  experiment=finetune/wmdp/forget \
  trainer="$trainer_config" \
  data_split="$domain" \
  task_name="$(basename "$output_dir")" \
  paths.output_dir="$output_dir" \
  model.model_args.pretrained_model_name_or_path="$source_model" \
  model.model_args.attn_implementation=eager \
  model.tokenizer_args.pretrained_model_name_or_path="$source_model"
