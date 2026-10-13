#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail
source scripts/slurm/environment.sh
"$EXPERIMENT_PYTHON" scripts/perta_mask.py "$@"
