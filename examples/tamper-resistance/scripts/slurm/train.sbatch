#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
mkdir -p artifacts/logs artifacts/data/hf-cache

# Compute nodes may inject Python 3.8 CVMFS paths that make the Python 3.12
# environment fail during interpreter initialization.
unset PYTHONPATH
source .venv-train/bin/activate
export AXOLOTL_DO_NOT_TRACK=1
export HF_DATASETS_CACHE="$PWD/artifacts/data/hf-cache"

config=${1:?usage: train.sbatch AXOLOTL_CONFIG}
prepared=artifacts/data/wmdp-bio-forget-text-v1
if [[ -f "$prepared/.lilpipe-text-v1" ]]; then
  echo "Reusing prepared corpus at $prepared"
else
  python scripts/data/prepare_forget_corpus.py \
    --output "$prepared" \
    --cache-dir "$HF_DATASETS_CACHE"
fi

axolotl train "$config" --launcher python
