#!/bin/bash
# Figure 2 attack sweep with BeaverTails attack indices 8000:9000.
#SBATCH --job-name=seam-disjoint-eval
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-4

set -euo pipefail

: "${SEAM_ROOT:?Set SEAM_ROOT to the upstream SEAM checkout}"
: "${SEAM_ENV:?Set SEAM_ENV to the prepared Python environment}"
: "${SEAM_RUN_ROOT:?Set SEAM_RUN_ROOT to the scratch run directory}"
: "${SEAM_MODEL:?Set SEAM_MODEL to the completed defense checkpoint}"

LEARNING_RATES=(2e-5 5e-5 8e-5 1e-4 2e-4)
LEARNING_RATE=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
PRE_UTILITY=False
if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    PRE_UTILITY=True
fi

export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRATCH/.cache/huggingface-seam-datasets-3.3.1}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR="$SEAM_RUN_ROOT/wandb"

mkdir -p "$WANDB_DIR" "$SEAM_RUN_ROOT/eval/lr-$LEARNING_RATE"
cd "$SEAM_ROOT"

CUDA_VISIBLE_DEVICES=0 "$SEAM_ENV/bin/python" -c '
import src.eval as seam_eval

original_construct_dataset = seam_eval.construct_beavertails_dataset
original_eval_utility = seam_eval.eval_utility

def construct_disjoint_attack_dataset(tokenizer, *args, **kwargs):
    # Materialize harmful indices 0:9000, then retain only the 1000 examples
    # outside the defense pool at indices 0:8000.
    kwargs["attack"] = True
    kwargs["attack_size"] = 9000
    harmful, harmless, test = original_construct_dataset(tokenizer, *args, **kwargs)
    return (
        harmful.select(range(8000, 9000)),
        harmless.select(range(8000, 9000)),
        test,
    )

def eval_utility(*args, **kwargs):
    results = original_eval_utility(*args, **kwargs)
    for task in ("arc_easy", "hellaswag"):
        results[task]["acc,none"] = results[task]["acc_norm,none"]
    return results

seam_eval.construct_beavertails_dataset = construct_disjoint_attack_dataset
seam_eval.eval_utility = eval_utility
seam_eval.eval_harmfulness_gpt = lambda *args, **kwargs: float("nan")
seam_eval.main()
' \
    --output_dir "$SEAM_RUN_ROOT/eval/lr-$LEARNING_RATE" \
    --model_name "$SEAM_MODEL" \
    --attack_size 1000 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --utility_tasks mmlu truthfulqa_mc1 arc_easy hellaswag \
    --pre_utility "$PRE_UTILITY" \
    --wandb_run_name "seam_disjoint_eval_indices8000-9000_lr$LEARNING_RATE"
