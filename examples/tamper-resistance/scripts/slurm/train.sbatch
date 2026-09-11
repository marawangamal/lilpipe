#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export PYTHONPATH="$PWD"
export WANDB_DIR="$PWD/artifacts/logs"

source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
axolotl train "$1" --launcher python
