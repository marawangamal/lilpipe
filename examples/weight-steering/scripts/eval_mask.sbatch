#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$project_dir"

base_model=$1
adapter=$2
merge_config=$3
output=$4
limit=${5:-100}
model=$base_model

source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"
axolotl_bin="$VIRTUAL_ENV/bin/axolotl"
if [[ "$adapter" != none ]]; then
  merge_dir="${SLURM_TMPDIR:-/tmp}/lilpipe-mask-${SLURM_JOB_ID:-local}"
  "$axolotl_bin" merge-lora "$merge_config" \
    --base-model "$base_model" \
    --lora-model-dir "$adapter" \
    --output-dir "$merge_dir"
  model="$merge_dir/merged"
fi

mkdir -p "$output"
source "$SCRATCH/lilpipe/examples/weight-steering/.venv-inspect/bin/activate"

judge_model=Qwen/Qwen3.6-27B-FP8
judge_port=18123
judge_log="artifacts/logs/mask-judge-${SLURM_JOB_ID:-local}.log"
vllm serve "$judge_model" \
  --host 127.0.0.1 \
  --port "$judge_port" \
  --served-model-name "$judge_model" \
  --dtype auto \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  >"$judge_log" 2>&1 &
judge_pid=$!
trap 'kill "$judge_pid" 2>/dev/null || true; wait "$judge_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:$judge_port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$judge_pid" 2>/dev/null; then
    echo "MASK judge exited during startup; see $judge_log" >&2
    exit 1
  fi
  sleep 5
done
curl --fail --silent "http://127.0.0.1:$judge_port/health" >/dev/null || {
  echo "MASK judge did not become healthy; see $judge_log" >&2
  exit 1
}

export LOCAL_JUDGE_API_KEY=local
export LOCAL_JUDGE_BASE_URL="http://127.0.0.1:$judge_port/v1"
inspect eval inspect_tasks/mask_fast.py@mask_fast \
  --model "vllm/$model" \
  -M dtype=float16 \
  -M max_model_len=8192 \
  -M max_num_seqs=8 \
  -M gpu_memory_utilization=0.25 \
  -M "default_chat_template_kwargs='{\"enable_thinking\": false}'" \
  --max-connections 8 \
  --sample-shuffle 42 \
  --limit "$limit" \
  --log-dir "$output" \
  --log-format json

python scripts/compute_mask_results.py "$output"
