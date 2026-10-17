#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%A_%a.out
#SBATCH --time=12:00:00
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export UV_CONCURRENT_INSTALLS=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export WANDB_DIR="$PWD/artifacts/mila/logs"
export WANDB_MODE=offline
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

attack=${1:?missing relearning kind}
method=${2:?missing method}
base_model=${3:?missing base model}
result_root=${4:?missing result root}
task=${SLURM_ARRAY_TASK_ID:?array task required}
seed=$((task % 3 + 1))

case "$attack" in
  harmful)
    learning_rates=(1e-5 5e-6 1e-6)
    lr=${learning_rates[$(((task / 3) % 3))]}
    fractions=(0.1 0.3)
    fraction=${fractions[$((task / 9))]}
    variant="p-${fraction}/lr-${lr}/seed-${seed}"
    extra=(--fraction "$fraction")
    ;;
  benign)
    learning_rates=(1e-5 1e-6 1e-7)
    lr=${learning_rates[$((task / 3))]}
    fraction=
    variant="lr-${lr}/seed-${seed}"
    extra=()
    ;;
  *) echo "unsupported relearning kind: $attack" >&2; exit 2 ;;
esac

template="configs/stages-tofu/2_relearn/${method}--relearn-fft/${attack}.yml"
[[ -f "$template" ]] || { echo "missing relearning config: $template" >&2; exit 2; }
model_dir="$result_root/$method/$attack/$variant/model"
eval_dir="$result_root/$method/$attack/$variant"

install_lock="$UV_CACHE_DIR/.lilpipe-install.lockdir"
until mkdir "$install_lock" 2>/dev/null; do sleep 2; done
trap 'rmdir "$install_lock"' EXIT
uv sync --offline --frozen --group train --group eval
rmdir "$install_lock"
trap - EXIT
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python scripts/run_tofu_relearn.py \
  "$template" "$base_model" "$model_dir" \
  --learning-rate "$lr" --seed "$seed" "${extra[@]}"
python scripts/evaluate_tofu_attack.py \
  "$model_dir" "$eval_dir" \
  --attack "$attack" --learning-rate "$lr" --seed "$seed" "${extra[@]}"

case "$model_dir" in
  artifacts/mila/attacks/tofu-table2/*/model) rm -rf -- "$model_dir" ;;
  *) echo "refusing to remove unexpected model path: $model_dir" >&2; exit 2 ;;
esac
