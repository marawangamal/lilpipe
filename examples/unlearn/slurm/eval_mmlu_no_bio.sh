#!/bin/bash
#SBATCH --job-name=eval-mmlu-no-bio
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/logs/eval-mmlu-no-bio-%j.out
set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the examples/unlearn directory}
cd "$repo_root"
[[ -f unlearn/__init__.py ]] || {
    echo "ERROR: submit from examples/unlearn: sbatch slurm/eval_mmlu_no_bio.sh ..." >&2
    exit 1
}

model_id=${1:?missing model id}
model_path=${2:?missing model path}
if [[ "$model_path" == artifacts/* || "$model_path" == /* ]] && \
   [[ ! -f "$model_path/config.json" ]]; then
    echo "ERROR: model not found at $model_path" >&2
    exit 1
fi

tamper_root=$(cd ../tamper-resistance && pwd)
output="artifacts/evals/$model_id/mmlu-no-bio"
model_args="pretrained=$model_path,dtype=bfloat16,trust_remote_code=True"

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-unlearn-eval"
export PYTHONPATH="$repo_root"

uv sync --project "$tamper_root" --frozen --group unlearn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

mkdir -p "$output"
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "$model_args" \
  --include_path "$tamper_root/lm_eval_tasks" \
  --tasks mmlu_no_bio \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
