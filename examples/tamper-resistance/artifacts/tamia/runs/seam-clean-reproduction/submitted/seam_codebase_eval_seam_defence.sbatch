#!/bin/bash
#SBATCH --job-name=seam-eval
#SBATCH --output=artifacts/tamia/logs/seam_eval-%j.out
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

# CONFIG RUN MODEL OUTPUT [MODEL OUTPUT ...]. Four independent upstream evaluators.
config=$(realpath "${1:?missing attack/eval config}")
run=$(realpath -m "${2:?missing run directory}")
shift 2
(( $# > 0 && $# % 2 == 0 ))
models=() outputs=()
while (( $# )); do
    models+=("$(realpath "$1")")
    outputs+=("$(realpath -m "$2")")
    shift 2
done
IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p "$run/wandb"
export WANDB_DIR="$run/wandb"
runner="$PWD/scripts/misc/run_seam_upstream.py"
root=$(realpath ../seam)
worker() {
    local gpu=$1 index
    for ((index=gpu; index<${#models[@]}; index+=4)); do
        mkdir -p "${outputs[index]}"
        CUDA_VISIBLE_DEVICES="${gpus[gpu]}" "$seam_env/bin/python" "$runner" \
          --mode eval --seam-root "$root" --report "${outputs[index]}/metrics.json" \
          --config "$config" --model_name "${models[index]}" --attack False \
          --pre_utility False --post_utility True \
          --utility_tasks mmlu truthfulqa_mc1 arc_easy hellaswag \
          > "${outputs[index]}/eval.log" 2>&1
    done
}
pids=()
for gpu in 0 1 2 3; do worker "$gpu" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
