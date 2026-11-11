#!/bin/bash
#SBATCH --job-name=seam-relearning-sweep
#SBATCH --output=artifacts/tamia/logs/seam_relearning_sweep-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/lilpipe/seam-clean/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/lilpipe/seam-clean/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline OMP_NUM_THREADS=8
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}

# CONFIG MODEL MODELS_ROOT RUN: upstream attack, with persistence-only hooks.
config=$(realpath "${1:?missing attack config}")
model=$(realpath "${2:?missing defended model}")
models_root=$(realpath "${3:?missing output root}")
run=$(realpath -m "${4:?missing run directory}")
base=$(basename "$model")
rates=(2e-5 5e-5 8e-5 1e-4 2e-4)
IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "$run/wandb"
export WANDB_DIR="$run/wandb"
runner="$PWD/scripts/misc/run_seam_upstream.py"
root=$(realpath ../seam)
for lr in "${rates[@]}"; do
    test ! -e "$models_root/$base--relearn-fft-lr$lr-ms250"
done
worker() {
    local gpu=$1 index lr name
    for ((index=gpu; index<${#rates[@]}; index+=4)); do
        lr=${rates[index]}
        name="$base--relearn-fft-lr$lr-ms250"
        CUDA_VISIBLE_DEVICES="${gpus[gpu]}" "$seam_env/bin/python" "$runner" \
          --mode attack --seam-root "$root" --report "$run/$name.json" \
          --config "$config" --model_name "$model" --output_dir "$models_root/$name" \
          --learning_rate "$lr" --wandb_run_name "$name" \
          --pre_utility False --post_utility False > "$run/$name.log" 2>&1
    done
}
pids=()
for gpu in 0 1 2 3; do worker "$gpu" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
