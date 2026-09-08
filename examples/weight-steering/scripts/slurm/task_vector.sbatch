#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"

python src/steering/task_vector.py "$1"
