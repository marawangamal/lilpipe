#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source .venv/bin/activate

export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
config=$1
axolotl train "$config" --launcher python
axolotl merge-lora "$config"
