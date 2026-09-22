#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

module load cuda/12.1.1

model_id=${1:?missing model id}
model_name_or_path=${2:?missing model name or path}
domain=${3:?missing WMDP domain}
utility_task=${4:-mmlu}
output_dir="artifacts/evals/$model_id"

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/.cache/huggingface/datasets"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-open-unlearning-$SLURM_JOB_ID"
export PYTHONPATH="$PWD"

uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
uv pip install --offline --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e '.[lm-eval]'
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

if [[ "$model_name_or_path" == "HuggingFaceH4/zephyr-7b-beta" ]]; then
  model_cache="$HF_HUB_CACHE/models--HuggingFaceH4--zephyr-7b-beta"
  revision=$(<"$model_cache/refs/main")
  model_name_or_path="$model_cache/snapshots/$revision"
fi
test -f "$model_name_or_path/config.json"

mkdir -p "$output_dir" artifacts/logs
python src/eval.py --config-name=eval.yaml \
  experiment=eval/wmdp/default \
  data_split="$domain" \
  task_name="$model_id" \
  model.model_args.pretrained_model_name_or_path="$model_name_or_path" \
  model.model_args.attn_implementation=eager \
  model.tokenizer_args.pretrained_model_name_or_path="$model_name_or_path" \
  eval.lm_eval.include_path=lm_eval_tasks \
  "eval.lm_eval.tasks=[wmdp_${domain},${utility_task}]" \
  eval.lm_eval.simple_evaluate_args.batch_size=4 \
  paths.output_dir="$output_dir"
