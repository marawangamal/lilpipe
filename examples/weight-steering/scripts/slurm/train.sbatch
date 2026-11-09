#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source scripts/slurm/activate_local_env.sh axolotl

export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/artifacts/data/hf-cache/${SLURM_JOB_ID:-local}}"
mkdir -p "$HF_DATASETS_CACHE"

config=$1
seed=${2:-}
output_dir=${3:-}
dataset_prepared_path=${4:-}
extra_overrides=("${@:5}")
overrides=()
if [[ -n "$seed" ]]; then
    overrides+=(--seed "$seed")
fi
if [[ -n "$output_dir" ]]; then
    overrides+=(--output-dir "$output_dir")
fi
if [[ -n "$dataset_prepared_path" ]]; then
    mkdir -p "$dataset_prepared_path"
    overrides+=(--dataset-prepared-path "$dataset_prepared_path")
fi
axolotl train "$config" "${overrides[@]}" "${extra_overrides[@]}" --launcher python
