#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail
source scripts/slurm/environment.sh
"$EXPERIMENT_PYTHON" scripts/proxy_eval.py --model "$1" --output "$2"
