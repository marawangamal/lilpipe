#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source scripts/slurm/activate_local_env.sh axolotl

base_model=$1
adapter=$2
output=$3
adapter_scale=${4:-1.0}
adapter_args=()
if [[ "$adapter" != none ]]; then
  adapter_args=(--adapter "$adapter")
fi

python scripts/evaluation/sandbagging.py \
  --base "$base_model" \
  "${adapter_args[@]}" \
  --adapter-scale "$adapter_scale" \
  --out "$output/results.json"
