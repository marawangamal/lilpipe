#!/bin/bash
#SBATCH --job-name=collect-cb-trajectory
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/logs/collect-two-stage-%j.out
set -euo pipefail

repo_root=${SLURM_SUBMIT_DIR:?Submit this job from the examples/unlearn directory}
cd "$repo_root"

python scripts/collect_two_stage_results.py \
  artifacts/evals/cb-ret10-two-stage \
  artifacts/results/cb-ret10-two-stage.json
