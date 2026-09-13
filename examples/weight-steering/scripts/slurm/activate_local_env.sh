#!/bin/bash

set -euo pipefail

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  echo "SLURM_TMPDIR is required for the node-local Python environment" >&2
  return 1
fi

dependency_group=${1:-axolotl}
runtime_dir="$SLURM_TMPDIR/lilpipe-${SLURM_JOB_ID:-local}"
export UV_PYTHON_INSTALL_DIR="$runtime_dir/uv-python"
export UV_CACHE_DIR="$runtime_dir/uv-cache"
export UV_PROJECT_ENVIRONMENT="$runtime_dir/.venv-$dependency_group"

uv python install 3.12 --no-progress
local_python=$(find "$UV_PYTHON_INSTALL_DIR" -type f -path '*/bin/python3.12' -print -quit)
if [[ -z "$local_python" ]]; then
  echo "uv did not install a node-local Python 3.12 interpreter" >&2
  return 1
fi
uv sync \
  --frozen \
  --only-group "$dependency_group" \
  --python "$local_python" \
  --link-mode copy \
  --no-progress
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
