#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AXOLOTL_DO_NOT_TRACK=1
axolotl train "$1" --launcher python
