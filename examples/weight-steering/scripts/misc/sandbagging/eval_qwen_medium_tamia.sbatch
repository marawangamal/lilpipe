#!/bin/bash
#SBATCH --job-name=eval-qwen-medium
#SBATCH --output=artifacts/logs/tamia-qwen-medium-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
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
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda-cache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$SLURM_TMPDIR/torchinductor-cache"
export MALLOC_ARENA_MAX=1
export CPATH="/usr/include:/usr/include/x86_64-linux-gnu${CPATH:+:$CPATH}"

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
export PYTHONPATH="$PWD/configs/training:$PWD${PYTHONPATH:+:$PYTHONPATH}"
base_model=artifacts/models/Qwen3.8-27B-base
adapter=artifacts/models/Qwen3.8-27B-SFT-WMDP-Cyber-Sandbagging-MO
export QWEN_TOKENIZER_PATH="$adapter"
output=artifacts/evals/Qwen3.8-27B-SFT-WMDP-Cyber-Sandbagging-MO/prompt-generalization-thinking-medium-tamia
merge_dir="$SLURM_TMPDIR/qwen-medium"
mkdir -p "$output" "$HF_DATASETS_CACHE"
test -f artifacts/data/wmdp-cyber-sandbagging/test.jsonl
uv sync --offline --frozen --group axolotl
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# Match the original thinking-off evaluation's NF4-simulated merged weights.
axolotl merge-lora configs/training/qwen38/wmdp-cyber-sandbagging.yml \
    --base-model "$base_model" --lora-model-dir "$adapter" --output-dir "$merge_dir"

eval_args=(
    --model hf-multimodal
    --model_args "pretrained=$merge_dir/merged,dtype=float16,trust_remote_code=True,max_length=32768,enable_thinking=true"
    --tasks wmdp_cyber_sandbagging_qwen_medium
    --include_path lm_eval_tasks
    --batch_size 1
    --gen_kwargs "max_gen_toks=8192,do_sample=False"
    --log_samples
    --confirm_run_unsafe_code
)

# Give every GPU a document during the smoke test (four questions per framing).
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=no -m lm_eval run \
    "${eval_args[@]}" --limit 4 --output_path "$output/smoke/results.json"
python - "$output/smoke" <<'PYTHON'
import json
import sys
from pathlib import Path

files = list(Path(sys.argv[1]).rglob("results*.json"))
assert files, "Smoke evaluation did not save metrics"
results = json.loads(files[-1].read_text())["results"]
valid = [v for row in results.values() for key, v in row.items()
         if key.startswith("valid_answer,")]
assert valid and max(valid) > 0, "Smoke produced no parseable final answers; inspect saved responses"
print("Smoke evaluation produced parseable final answers.", flush=True)
PYTHON

# Each process evaluates a different subset on one H100.
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=no -m lm_eval run \
    "${eval_args[@]}" --output_path "$output/results.json" "$@"
