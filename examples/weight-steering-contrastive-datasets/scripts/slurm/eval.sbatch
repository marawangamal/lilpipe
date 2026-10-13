#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail
source scripts/slurm/environment.sh
cell=${1:?missing cell}
model=${2:?missing model}
oracle=${3:?missing oracle}
"$EXPERIMENT_PYTHON" ../weight-steering/scripts/evaluation/tofu.py "$model" none none "artifacts/evals/$cell" "$oracle" --forget-split forget10 --retain-split retain90
