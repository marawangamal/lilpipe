#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%j.out
#SBATCH --time=03:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_INSTALLS=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

model_id=${1:?missing model id}
baseline_model=${2:?missing baseline model}
trajectory=${3:?missing attack trajectory}
learning_rate=${4:?missing learning rate}
attack_data=${5:?missing attack data role}
method=${6:?missing method}
result_root=${7:?missing result root}
checkpoint_interval=${8:?missing checkpoint interval}
max_steps=${9:?missing maximum step}

mkdir -p "$UV_CACHE_DIR"
install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group eval
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
for ((step = 0; step <= max_steps; step += checkpoint_interval)); do
    python scripts/evaluate_beavertail_and_capabilities.py trajectory \
        "$model_id" \
        "$baseline_model" \
        "$trajectory" \
        "$learning_rate" \
        "$attack_data" \
        "$method" \
        "$result_root" \
        --checkpoint-step "$step"
done

bash scripts/slurm/tamia/cleanup_beavertail_trajectory.sbatch \
    "$model_id" \
    "$trajectory" \
    "$result_root" \
    "$checkpoint_interval" \
    "$max_steps"
