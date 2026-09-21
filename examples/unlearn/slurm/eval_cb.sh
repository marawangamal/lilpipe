#!/bin/bash
#SBATCH --job-name=eval-unlearn-cb
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/eval-cb-%j.out
set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the examples/unlearn directory}
cd "$repo_root"
[[ -f unlearn/__init__.py ]] || {
    echo "ERROR: submit from examples/unlearn: sbatch slurm/eval_cb.sh ..." >&2
    exit 1
}

model_id=${1:?missing model id}
model_path=${2:?missing model path}
[[ -f "$model_path/config.json" ]] || {
    echo "ERROR: model not found at $model_path" >&2
    exit 1
}

tamper_root=$(cd ../tamper-resistance && pwd)
output_root="artifacts/evals/$model_id"
model_args="pretrained=$model_path,dtype=bfloat16,trust_remote_code=True"

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-unlearn-eval"
export PYTHONPATH="$repo_root"

uv sync --project "$tamper_root" --frozen --group unlearn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

mkdir -p "$output_root/wmdp-bio-robust" "$output_root/mmlu"

accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "$model_args" \
  --include_path unlearn/lm_eval_tasks \
  --tasks wmdp_bio_robust \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output_root/wmdp-bio-robust"

accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "$model_args" \
  --tasks mmlu \
  --num_fewshot 1 \
  --batch_size 32 \
  --output_path "$output_root/mmlu"
