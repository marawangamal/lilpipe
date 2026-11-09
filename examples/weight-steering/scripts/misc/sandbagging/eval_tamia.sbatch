#!/bin/bash
#SBATCH --job-name=eval-granite-sandbagging
#SBATCH --output=artifacts/logs/tamia-eval-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export CPATH="/usr/include:/usr/include/x86_64-linux-gnu${CPATH:+:$CPATH}"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda-cache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor-cache"
export MALLOC_ARENA_MAX=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"

adapter=artifacts/models/Granite-4.2-30B-SFT-WMDP-Cyber-Paired-Sandbagging-MO-Tamia
output=artifacts/evals/Granite-4.2-30B-SFT-WMDP-Cyber-Paired-Sandbagging-MO-Tamia/prompt-generalization-merged-b8
merge_dir="$SLURM_TMPDIR/merged"
test_data=artifacts/data/wmdp-cyber-sandbagging/test.jsonl

mkdir -p artifacts/logs "$HF_DATASETS_CACHE" "$output"
test -f "$test_data"
uv sync --frozen --group axolotl
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

axolotl merge-lora \
    configs/training/granite42/wmdp-cyber-paired-sandbagging-tamia.yml \
    --base-model ibm-granite/granite-4.2-30b \
    --lora-model-dir "$adapter" \
    --output-dir "$merge_dir"

accelerate launch --num_processes=1 -m lm_eval run \
    --model hf \
    --model_args "pretrained=$merge_dir/merged,dtype=float16,trust_remote_code=True,enable_thinking=false" \
    --tasks wmdp_cyber_sandbagging_granite42 \
    --include_path lm_eval_tasks \
    --batch_size 8 \
    --gen_kwargs "max_gen_toks=16,do_sample=False" \
    --output_path "$output/results.json" \
    --log_samples \
    --confirm_run_unsafe_code
