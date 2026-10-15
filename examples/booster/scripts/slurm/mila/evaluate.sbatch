#!/bin/bash
#SBATCH --output=artifacts/mila/logs/slurm-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python_bin="$SCRATCH/lilpipe/envs/booster-figure3/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing prepared environment: $python_bin" >&2
    exit 2
fi

method=${1:?missing method}
attack=${2:?missing attack}
defense_adapter=${3:?missing defense adapter}
attack_dir=${4:?missing attack directory}
output_dir=${5:?missing output directory}
attack_lora_init=${6:-continued}
step=$((SLURM_ARRAY_TASK_ID * 200))

mkdir -p artifacts/mila/logs

child=""
requeue() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    scontrol requeue "$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"
    exit 0
}
trap requeue USR1

"$python_bin" scripts/lilpipe/evaluate_figure3.py \
    --method "$method" \
    --attack "$attack" \
    --attack-lora-init "$attack_lora_init" \
    --step "$step" \
    --defense-adapter "$defense_adapter" \
    --attack-dir "$attack_dir" \
    --output-dir "$output_dir" &
child=$!
wait "$child"
