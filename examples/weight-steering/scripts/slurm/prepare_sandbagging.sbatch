#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
if [[ ! -s artifacts/data/sandbagging/train.jsonl || \
      ! -s artifacts/data/sandbagging/test.jsonl ]]; then
  echo "Sandbagging task pool is missing." >&2
  echo "Build and verify it on the networked login node first:" >&2
  echo "  python scripts/data/build_sandbagging_tasks.py" >&2
  exit 1
fi
