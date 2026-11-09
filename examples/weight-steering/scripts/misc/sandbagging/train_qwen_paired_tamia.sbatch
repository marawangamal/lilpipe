#!/bin/bash
#SBATCH --job-name=train-qwen-paired
#SBATCH --output=artifacts/logs/tamia-qwen-paired-%j.out
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=12:00:00
set -euo pipefail
mode=${1:?Expected lora or full}
phase=${2:-train}
revision=${3:-}
resume_checkpoint=${4:-}
[[ "$mode" == lora || "$mode" == full ]]
[[ "$phase" == smoke || "$phase" == train ]]
[[ -z "$revision" || "$revision" == v2 || "$revision" == v3 ]]
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf-datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda-cache"
export TRITON_CACHE_DIR="$SCRATCH/.cache/qwen-paired-training-triton"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor-cache"
export WANDB_MODE=offline WANDB_DIR="$SLURM_TMPDIR/wandb"
export AXOLOTL_DO_NOT_TRACK=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CPATH="/usr/include:/usr/include/x86_64-linux-gnu${CPATH:+:$CPATH}"
source "$SCRATCH/lilpipe-sandbagging-tamia/envs/axolotl-paired/bin/activate"
suffix=${revision:+-$revision}
config="configs/training/qwen38/wmdp-cyber-paired-$mode$suffix-tamia.yml"
output="artifacts/models/Qwen3.8-27B-SFT-WMDP-Cyber-Paired-${mode^}-MO$suffix"
overrides=(--save-strategy steps --save-steps 50 --save-total-limit 1)
if [[ "$phase" == smoke ]]; then
    output="artifacts/models/qwen-paired-smoke-$mode$suffix"
    overrides+=(--max-steps 1 --save-steps 1)
fi
if [[ -n "$resume_checkpoint" ]]; then
    [[ "$phase" == train ]]
    test -f "$resume_checkpoint/trainer_state.json"
    [[ -f "$resume_checkpoint/optimizer.bin" || -f "$resume_checkpoint/optimizer.pt" ]]
    overrides+=(--resume-from-checkpoint "$resume_checkpoint")
fi
mkdir -p "$output" "$WANDB_DIR" "$HF_DATASETS_CACHE"
axolotl train "$config" --launcher accelerate --num-processes 4 \
    --output-dir "$output" "${overrides[@]}"
python - "$output" "$phase" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
checkpoints = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
assert checkpoints, "No resumable training checkpoint saved"
state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
assert state["global_step"] >= 1
assert any("optimizer" in p.name for p in checkpoints[-1].rglob("*")), "Optimizer state missing"
print("Checkpoint and optimizer state verified:", state["global_step"], flush=True)
PY
if [[ "$phase" == train && "$mode" == lora ]]; then
    CUDA_VISIBLE_DEVICES=0 axolotl merge-lora "$config" \
        --lora-model-dir "$output" --output-dir "$output"
fi
