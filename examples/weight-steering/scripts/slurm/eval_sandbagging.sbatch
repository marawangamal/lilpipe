#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source scripts/slurm/activate_local_env.sh axolotl

base_model=$1
adapter=$2
output=$3
model=$base_model
adapter_args=()
if [[ "$adapter" != none ]]; then
  adapter_args=(--adapter "$adapter")
fi

mkdir -p "$output"
python scripts/evaluation/sandbagging.py \
  --base "$model" \
  "${adapter_args[@]}" \
  --out "$output/results.json" \
  --limit-per-domain 25 \
  --batch-size 8 \
  --max-new-tokens 64
