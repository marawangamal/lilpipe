#!/bin/bash
#SBATCH --job-name=seam-defence
#SBATCH --output=artifacts/tamia/logs/seam_defence-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/lilpipe/seam-clean/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/lilpipe/seam-clean/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}

# Run from examples/tamper-resistance: CONFIG MODEL RUN [UPSTREAM_ARGS...].
config=$(realpath "${1:?missing upstream config}")
model=$(realpath -m "${2:?missing output model}")
run=$(realpath -m "${3:?missing run directory}")
shift 3
test ! -e "$model"
mkdir -p "$run/wandb"
export WANDB_DIR="$run/wandb"
runner="$PWD/scripts/misc/run_seam_upstream.py"
root=$(realpath ../seam)

# Exercise one optimizer step and saving before the full, fresh-model run.
smoke="$SLURM_TMPDIR/seam-smoke"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$seam_env/bin/python" "$runner" \
  --mode defense --seam-root "$root" --report "$run/smoke.json" \
  --config "$config" --output_dir "$smoke" --max_steps 1 --save_steps 1
rm -rf -- "$smoke"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$seam_env/bin/python" "$runner" \
  --mode defense --seam-root "$root" --report "$run/defense.json" \
  --config "$config" --output_dir "$model" --wandb_run_name "$(basename "$model")" "$@"
# Keep one final defended model; intermediate checkpoints are unnecessary here.
find "$model" -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf -- {} +
