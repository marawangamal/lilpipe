#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

project_dir=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$project_dir"
source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"
axolotl_bin="$VIRTUAL_ENV/bin/axolotl"

base_model=$1
adapter=$2
merge_config=$3
output=$4
tasks=$5
adapter_scale=${6:-1.0}
model_backend=${7:-hf}
batch_size=${8:-32}
max_gen_toks=${9:-512}
apply_chat_template=${10:-true}
load_in_4bit=${11:-false}
enable_thinking=${12:-false}
limit=${13:-none}
direct_peft=${14:-false}
model=$base_model

if [[ "$adapter" != none && "$direct_peft" == true ]]; then
  if [[ "$adapter_scale" != 1 && "$adapter_scale" != 1.0 ]]; then
    echo "direct PEFT evaluation does not support adapter scaling" >&2
    exit 2
  fi
elif [[ "$adapter" != none ]]; then
  merge_dir="${SLURM_TMPDIR:-/tmp}/lilpipe-${SLURM_JOB_ID:-local}"
  if [[ "$adapter_scale" != 1 && "$adapter_scale" != 1.0 ]]; then
    scaled_adapter="$merge_dir/scaled-adapter"
    mkdir -p "$merge_dir"
    cp -a "$adapter" "$scaled_adapter"
    python - "$scaled_adapter/adapter_config.json" "$adapter_scale" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
config["lora_alpha"] *= float(sys.argv[2])
path.write_text(json.dumps(config, indent=2) + "\n")
PY
    adapter=$scaled_adapter
  fi
  "$axolotl_bin" merge-lora "$merge_config" \
    --base-model "$base_model" \
    --lora-model-dir "$adapter" \
    --output-dir "$merge_dir"
  model="$merge_dir/merged"
fi

mkdir -p "$output"
source "$SCRATCH/lilpipe/examples/weight-steering/.venv-lmeval/bin/activate"
lmeval_bundle=artifacts/env/lmeval-packages.tar
local_site="${SLURM_TMPDIR:-/tmp}/lilpipe-lmeval-${SLURM_JOB_ID:-local}/site-packages"
mkdir -p "$local_site"
tar -xf "$lmeval_bundle" -C "$local_site"
# Reuse bitsandbytes from the Axolotl environment for optional NF4 evaluation.
axolotl_site="$project_dir/.venv/lib/python3.12/site-packages"
export PYTHONPATH="$local_site:$axolotl_site${PYTHONPATH:+:$PYTHONPATH}"
# lm-eval's multimodal backend currently drops HFLM chat-template arguments.
# Forward enable_thinking=False to the processor so Qwen closes the think block.
sed -i \
  '/continue_final_message=not add_generation_prompt,/a\            **self.chat_template_args,' \
  "$local_site/lm_eval/models/hf_vlms.py"
# Transformers 5 requires a quantization config instead of the legacy
# load_in_4bit model kwarg still emitted by lm-eval.
sed -i \
  '/            self._model = self.AUTO_MODEL_CLASS.from_pretrained(/i\            if model_kwargs.pop("load_in_4bit", False):\
                quantization_config = transformers.BitsAndBytesConfig(\
                    load_in_4bit=True,\
                    bnb_4bit_quant_type="nf4",\
                    bnb_4bit_compute_dtype=get_dtype(dtype),\
                )' \
  "$local_site/lm_eval/models/huggingface.py"
chat_template_args=()
if [[ "$apply_chat_template" == true ]]; then
  chat_template_args+=(--apply_chat_template)
fi
model_args="pretrained=$model,dtype=float16,trust_remote_code=True,enable_thinking=$enable_thinking"
if [[ "$load_in_4bit" == true ]]; then
  model_args+=",load_in_4bit=True"
fi
if [[ "$adapter" != none && "$direct_peft" == true ]]; then
  model_args+=",peft=$adapter"
fi
limit_args=()
if [[ "$limit" != none ]]; then
  limit_args+=(--limit "$limit")
fi
accelerate launch --num_processes=1 -m lm_eval run \
  --model "$model_backend" \
  --model_args "$model_args" \
  --tasks "$tasks" \
  --include_path lm_eval_tasks \
  --batch_size "$batch_size" \
  "${chat_template_args[@]}" \
  "${limit_args[@]}" \
  --gen_kwargs "max_gen_toks=$max_gen_toks,do_sample=False" \
  --output_path "$output/results.json" \
  --log_samples \
  --confirm_run_unsafe_code
