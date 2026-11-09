#!/bin/bash
#SBATCH --job-name=train-granite-sandbagging
#SBATCH --output=artifacts/logs/tamia-train-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

project_dir=$PWD
local_log="$SLURM_TMPDIR/train.log"
shared_log="$project_dir/artifacts/logs/tamia-train-$SLURM_JOB_ID.out"
trap 'cp "$local_log" "$shared_log" 2>/dev/null || true' EXIT
exec >"$local_log" 2>&1

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
shared_hf_home="$SCRATCH/.cache/huggingface"
export HF_HOME="$SLURM_TMPDIR/huggingface"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf-datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export CPATH="/usr/include:/usr/include/x86_64-linux-gnu${CPATH:+:$CPATH}"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda-cache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor-cache"
export WANDB_MODE=offline
export WANDB_DIR="$SLURM_TMPDIR/wandb"
local_project="$SLURM_TMPDIR/project"
local_output="$SLURM_TMPDIR/model"
shared_output="$project_dir/artifacts/models/Granite-4.2-30B-SFT-WMDP-Cyber-Paired-Sandbagging-MO-Tamia"

mkdir -p \
    artifacts/logs \
    "$XDG_CACHE_HOME" \
    "$HF_HOME/hub" \
    "$HF_DATASETS_CACHE" \
    "$CUDA_CACHE_PATH" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$WANDB_DIR"
uv sync --frozen --group axolotl
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

rsync -a --exclude='*.incomplete' \
    "$shared_hf_home/hub/models--ibm-granite--granite-4.2-30b/" \
    "$HF_HOME/hub/models--ibm-granite--granite-4.2-30b/"

mkdir -p \
    "$local_project/configs/training/granite42" \
    "$local_project/artifacts/data/wmdp-cyber-sandbagging"
cp -a "$project_dir/scripts" "$project_dir/lm_eval_tasks" "$local_project/"
cp -a \
    "$project_dir/configs/training/granite42/wmdp-cyber-paired-sandbagging-tamia.yml" \
    "$local_project/configs/training/granite42/"
cp -a \
    "$project_dir/artifacts/data/wmdp-cyber-sandbagging/train-weak.jsonl" \
    "$local_project/artifacts/data/wmdp-cyber-sandbagging/"

cd "$local_project"
export PYTHONPATH="$local_project/configs/training:$local_project${PYTHONPATH:+:$PYTHONPATH}"

axolotl train \
    configs/training/granite42/wmdp-cyber-paired-sandbagging-tamia.yml \
    --launcher accelerate \
    --num-processes 4 \
    --output-dir "$local_output"

mkdir -p "$shared_output"
rsync -a "$local_output/" "$shared_output/"
