#!/bin/bash
#SBATCH --output=artifacts/logs/slurm-%j.out
set -euo pipefail

module load cuda/12.1.1

method=${1:?missing unlearning method}
domain=${2:?missing WMDP domain}
output_dir=${3:?missing output directory}

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/.cache/huggingface/datasets"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-open-unlearning-$SLURM_JOB_ID"
export PYTHONPATH="$PWD"
export WANDB_MODE="${WANDB_MODE:-disabled}"

uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
uv pip install --offline --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e '.[lm-eval]'
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

model_cache="$HF_HUB_CACHE/models--HuggingFaceH4--zephyr-7b-beta"
revision=$(<"$model_cache/refs/main")
model_snapshot="$model_cache/snapshots/$revision"
test -f "$model_snapshot/config.json"

forget_overrides=()
if [[ "$domain" == "bio" ]]; then
  bio_cache="$HF_HUB_CACHE/datasets--cais--wmdp-bio-forget-corpus"
  bio_revision=$(<"$bio_cache/refs/main")
  bio_files=("$bio_cache/snapshots/$bio_revision/data/"*.parquet)
  [[ -f "${bio_files[0]}" ]]
  bio_data_files=$(IFS=,; echo "[${bio_files[*]}]")
  forget_overrides+=(
    data.forget.WMDP_forget.args.hf_args.path=parquet
    "data.forget.WMDP_forget.args.hf_args.data_files=$bio_data_files"
  )
else
  test -f "data/wmdp/wmdp-corpora/${domain}-forget-corpus.jsonl" || {
    echo "Missing WMDP ${domain} corpus; run: .venv/bin/python setup_data.py --wmdp" >&2
    exit 1
  }
fi

mkdir -p "$output_dir" artifacts/logs
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/wmdp/default \
  trainer="$method" \
  data_split="$domain" \
  task_name="$(basename "$output_dir")" \
  paths.output_dir="$output_dir" \
  model.model_args.pretrained_model_name_or_path="$model_snapshot" \
  model.model_args.attn_implementation=eager \
  model.tokenizer_args.pretrained_model_name_or_path="$model_snapshot" \
  trainer.args.do_eval=false \
  trainer.args.eval_strategy=no \
  trainer.args.report_to=none \
  "${forget_overrides[@]}"
