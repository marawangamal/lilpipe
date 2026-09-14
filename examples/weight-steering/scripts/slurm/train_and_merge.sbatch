#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source scripts/slurm/activate_local_env.sh axolotl

export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
config=$1
mode=${2:-lora}
axolotl train "$config" --launcher python
if [[ "$mode" == lora ]]; then
  axolotl merge-lora "$config"
elif [[ "$mode" != full ]]; then
  echo "Unknown training mode: $mode" >&2
  exit 2
fi
