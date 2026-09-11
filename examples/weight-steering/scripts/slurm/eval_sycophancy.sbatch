#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"

base_model=$1
adapter=$2
merge_config=$3
output=$4
model=$base_model

source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"
axolotl_bin="$VIRTUAL_ENV/bin/axolotl"
if [[ "$adapter" != none ]]; then
  merge_dir="${SLURM_TMPDIR:-/tmp}/lilpipe-sycophancy-${SLURM_JOB_ID:-local}"
  "$axolotl_bin" merge-lora "$merge_config" \
    --base-model "$base_model" \
    --lora-model-dir "$adapter" \
    --output-dir "$merge_dir"
  model="$merge_dir/merged"
fi

mkdir -p "$output"
source "$SCRATCH/lilpipe/examples/weight-steering/.venv-sycophancy/bin/activate"
inspect_site="$VIRTUAL_ENV/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$inspect_site/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

judge_model=Qwen/Qwen3.6-27B-FP8
judge_port=18124
judge_log="artifacts/logs/sycophancy-judge-${SLURM_JOB_ID:-local}.log"
vllm serve "$judge_model" \
  --host 127.0.0.1 \
  --port "$judge_port" \
  --served-model-name "$judge_model" \
  --dtype auto \
  --gpu-memory-utilization 0.70 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --attention-backend FLASH_ATTN \
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
    echo "Sycophancy judge exited during startup; see $judge_log" >&2
    exit 1
  fi
  sleep 5
done
curl --fail --silent "http://127.0.0.1:$judge_port/health" >/dev/null || {
  echo "Sycophancy judge did not become healthy; see $judge_log" >&2
  exit 1
}

export LOCAL_JUDGE_API_KEY=local
export LOCAL_JUDGE_BASE_URL="http://127.0.0.1:$judge_port/v1"
inspect eval inspect_tasks/sycophancy.py@sycophancy \
  --model "vllm/$model" \
  -M dtype=float16 \
  -M max_model_len=8192 \
  -M max_num_seqs=8 \
  -M gpu_memory_utilization=0.25 \
  -M attention_backend=FLASH_ATTN \
  -M "default_chat_template_kwargs='{\"enable_thinking\": false}'" \
  --max-connections 8 \
  --log-dir "$output" \
  --log-format json

python scripts/evaluation/compute_sycophancy_results.py "$output"
