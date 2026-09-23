#!/bin/bash
#SBATCH --job-name=eval-cb-trajectory
#SBATCH --array=0-7
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/logs/eval-two-stage-%A_%a.out
set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the examples/unlearn directory}
cd "$repo_root"
[[ -f unlearn/__init__.py ]] || {
    echo "ERROR: submit from examples/unlearn" >&2
    exit 1
}

stage=${1:?missing stage}
base_model=${2:?missing base model}
adapter_dir=${3:?missing adapter directory}
eval_dir=${4:?missing evaluation directory}
final_kind=${5:?missing final kind: merged or adapter}

task_id=${SLURM_ARRAY_TASK_ID:?this script must run as an array task}
if [[ "$task_id" == 7 ]]; then
    step=32
else
    step=$((task_id * 5))
fi

tamper_root=$(cd ../tamper-resistance && pwd)
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-unlearn-eval"
export PYTHONPATH="$repo_root"

uv sync --project "$tamper_root" --frozen --group unlearn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

if [[ "$step" == 0 ]]; then
    model_args="pretrained=$base_model,dtype=bfloat16,trust_remote_code=True"
elif [[ "$step" == 32 && "$final_kind" == merged ]]; then
    model_args="pretrained=$adapter_dir,dtype=bfloat16,trust_remote_code=True"
else
    if [[ "$step" == 32 ]]; then
        checkpoint=$adapter_dir
    else
        checkpoint="$adapter_dir/checkpoint-$step"
    fi
    [[ -f "$checkpoint/adapter_config.json" ]] || {
        echo "ERROR: missing LoRA checkpoint $checkpoint" >&2
        exit 1
    }
    model_args="pretrained=$base_model,peft=$checkpoint,dtype=bfloat16,trust_remote_code=True"
fi

output_root="$eval_dir/$stage/checkpoint-$step"
mkdir -p "$output_root/wmdp-bio-robust" "$output_root/mmlu-no-bio"

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
  --include_path "$tamper_root/lm_eval_tasks" \
  --tasks mmlu_no_bio \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output_root/mmlu-no-bio"
