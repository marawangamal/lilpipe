#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"
export HF_DATASETS_CACHE="$PWD/artifacts/data/hf-cache/${SLURM_JOB_ID:-local}"
mkdir -p "$HF_DATASETS_CACHE" "$4"
python scripts/evaluation/tofu.py "$1" "$2" "$3" "$4" "$5"
