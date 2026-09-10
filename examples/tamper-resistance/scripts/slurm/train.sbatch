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

# Transformers 5 scans every model module during import. Spawned Hugging Face
# Datasets workers have intermittently observed files as missing when that scan
# reads the package from network scratch, so stage this small package locally.
scratch_site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [[ -n ${SLURM_TMPDIR:-} ]]; then
  local_root=$SLURM_TMPDIR/lilpipe-python
else
  local_root=$(mktemp -d /tmp/lilpipe-python-XXXXXX)
  trap 'rm -rf -- "$local_root"' EXIT
fi
local_site=$local_root/site-packages
mkdir -p "$local_site"
cp -a "$scratch_site/transformers" "$local_site/"
export PYTHONPATH="$local_site${PYTHONPATH:+:$PYTHONPATH}"
export LILPIPE_LOCAL_SITE=$local_site
python -c 'import os, pathlib, transformers; path = pathlib.Path(transformers.__file__).resolve(); root = pathlib.Path(os.environ["LILPIPE_LOCAL_SITE"]).resolve(); print(f"Using node-local Transformers: {path}"); assert path.is_relative_to(root)'

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
