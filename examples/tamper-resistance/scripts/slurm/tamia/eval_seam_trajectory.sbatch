#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

model_id=${1:?missing model id}
baseline_model=${2:?missing baseline model}
trajectory=${3:?missing attack trajectory}
learning_rate=${4:?missing learning rate}
attack_data=${5:?missing attack data role}
result_root=${6:?missing result root}
checkpoint_interval=${7:?missing checkpoint interval}
max_steps=${8:?missing maximum step}
array_index=${SLURM_ARRAY_TASK_ID:?this script must run as an array task}
gpus_per_task=${SLURM_GPUS_ON_NODE:?missing allocated GPUs}

module load httpproxy/1.0
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

pids=()
steps=()
for ((gpu = 0; gpu < gpus_per_task; gpu++)); do
    point=$((array_index * gpus_per_task + gpu))
    step=$((point * checkpoint_interval))
    if ((step > max_steps)); then
        continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu python scripts/evaluate_seam_figure2.py \
        "$model_id" \
        "$baseline_model" \
        "$trajectory" \
        "$learning_rate" \
        "$attack_data" \
        "$result_root" \
        --checkpoint-step "$step" &
    pids+=("$!")
    steps+=("$step")
done

status=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "Evaluation failed for checkpoint-${steps[$index]}" >&2
        status=1
    fi
done
exit "$status"
