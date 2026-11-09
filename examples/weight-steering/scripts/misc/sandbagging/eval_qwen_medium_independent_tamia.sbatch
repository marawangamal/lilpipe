#!/bin/bash
#SBATCH --job-name=eval-qwen-medium-independent
#SBATCH --output=artifacts/logs/tamia-qwen-medium-independent-%j.out
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

# Reuse the exact NF4-simulated merge preserved from the first attempt.
merged_model=artifacts/models/Qwen3.8-27B-SFT-WMDP-Cyber-Sandbagging-MO-merged-eval
test -f "$merged_model/model.safetensors.index.json"
eval_args=(
    --model hf-multimodal
    --model_args "pretrained=$merged_model,dtype=float16,trust_remote_code=True,max_length=32768,enable_thinking=true"
    --include_path lm_eval_tasks
    --gen_kwargs "max_gen_toks=8192,do_sample=False"
    --log_samples
    --confirm_run_unsafe_code
)

# A single-GPU smoke test has no distributed collectives or empty ranks.
CUDA_VISIBLE_DEVICES=0 python -m lm_eval run "${eval_args[@]}" \
    --tasks wmdp_cyber_sandbagging_qwen_medium --batch_size 1 --limit 1 \
    --use_cache "$output/independent-smoke-cache" \
    --output_path "$output/independent-smoke/results.json"
python - "$output/independent-smoke" <<'PYTHON'
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

# Disjoint task groups cover all seven conditions. Each process owns one GPU.
# Persistent response caches retain completed generations if a job is interrupted.
task_groups=(
    wmdp_cyber_sandbagging_qwen_medium_neutral,wmdp_cyber_sandbagging_qwen_medium_p4
    wmdp_cyber_sandbagging_qwen_medium_p1,wmdp_cyber_sandbagging_qwen_medium_p5
    wmdp_cyber_sandbagging_qwen_medium_p2,wmdp_cyber_sandbagging_qwen_medium_p6_unseen
    wmdp_cyber_sandbagging_qwen_medium_p3
)
pids=()
for gpu in 0 1 2 3; do
    worker_output="$output/worker-$gpu"
    mkdir -p "$worker_output"
    CUDA_VISIBLE_DEVICES="$gpu" python -m lm_eval run "${eval_args[@]}" \
        --tasks "${task_groups[$gpu]}" --batch_size 4 \
        --use_cache "$output/response-cache-$gpu" \
        --output_path "$worker_output/results.json" \
        >"$worker_output/worker.log" 2>&1 &
    pids+=("$!")
    echo "Started GPU $gpu worker ${pids[-1]}: ${task_groups[$gpu]}"
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
