#!/bin/bash
#SBATCH --output=artifacts/tamia/logs/slurm-%A_%a.out
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface-datasets-unlearn-smooth"
export PATH="$HOME/.local/bin:$PATH"

authors_root=${1:?missing authors WMDP root}
model_source=${2:?missing unlearned model path}
model_slug=${3:?missing artifact slug}
authors_root=$(realpath "$authors_root")
model_source=$(realpath "$model_source")

steps=(5 10 15)
example_counts=(20 40 60)
task=${SLURM_ARRAY_TASK_ID:?this script must run as an array task}
max_steps=${steps[$task]}
examples=${example_counts[$task]}
model_root="$authors_root/artifacts/tamia/models/$model_slug"
run_name="examples-$examples"
run_root="$model_root/$run_name"

if [[ ! -f "$model_source/config.json" ]]; then
  echo "Missing unlearned model: $model_source" >&2
  exit 1
fi
if [[ -e "$run_root" || -L "$run_root" ]]; then
  echo "Refusing existing relearning output: $run_root" >&2
  exit 1
fi

mkdir -p "$authors_root/artifacts/tamia/logs" "$model_root"
mkdir -p "$SLURM_TMPDIR/relearn-run"
ln -s "$SLURM_TMPDIR/relearn-run" "$run_root"

cd "$authors_root"
source "$SCRATCH/lilpipe/envs/unlearn-smooth-wmdp/bin/activate"
python src/exec/relearn_model.py \
  --config-file configs/unlearn/wmdp/Relearn+Forget.json \
  --overall.model_name "$model_source" \
  --overall.cache_dir "$HF_HOME/hub" \
  --overall.seed 1 \
  --dataset.dataset_seed 42 \
  --unlearn.lr 1e-5 \
  --unlearn.max_steps "$max_steps" \
  --logger.name "$run_name" \
  --logger.json.root "$model_root"

# Replace the temporary symlink with the complete checkpoint on shared scratch.
staged_run=$(realpath "$run_root")
unlink "$run_root"
mkdir -p "$run_root"
cp -a "$staged_run"/. "$run_root"/
