#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source scripts/slurm/activate_local_env.sh axolotl

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ ! -s artifacts/data/sandbagging/train.jsonl || \
      ! -s artifacts/data/sandbagging/test.jsonl ]]; then
  echo "Sandbagging task pool is missing." >&2
  echo "Build it on the networked login node first:" >&2
  echo "  python scripts/data/build_sandbagging_tasks.py" >&2
  exit 1
fi

base_model=$1
output=$2
python scripts/data/build_sandbagging_sft.py --base "$base_model" --out "$output"
