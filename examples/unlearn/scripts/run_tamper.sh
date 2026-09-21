#!/bin/bash
# Launch tamper (finetune) attack jobs on an unlearned model.
# By default submits 5 parallel sbatch jobs sweeping LR, dtype, and schedule:
#   fp16/linear at lr=1e-5, 2e-5, 8e-5  +  bf16/linear at lr=2e-5  +  fp16/cosine at lr=2e-5
#
# Usage:
#   bash scripts/run_tamper.sh -m models/EleutherAI/deep-ignorance-unfiltered_cb_sft_ret0_rm23_orth10_lr1e-3
#   bash scripts/run_tamper.sh -m models/EleutherAI/my_model --lr 2e-5
#   bash scripts/run_tamper.sh -m models/EleutherAI/my_model --lr 1e-5,2e-5,1e-4 --sched cosine
#
# Options:
#   --model, -m       Path to unlearned model (required)
#   --lr              Learning rate(s), comma-separated (default: see sweep below)
#   --steps           Max training steps (default: 10000)
#   --eval_every      Evaluate every N steps (default: 500)
#   --bs              Per-device batch size (default: 1)
#   --grad_acc        Gradient accumulation steps (default: 4; effective batch = bs * grad_acc * 4 GPUs)
#   --epochs          Number of epochs (default: 2)
#   --data            Tamper data: bio_remove, benign, bio_chat, bio_forget_flagged,
#                     bio_forget, flagged, wikitext, annealing (default: bio_remove)
#   --lora            LoRA rank for tamper attack (0 = full finetune, default: 0)
#   --lora_target     LoRA target modules: all, attn, mlp (default: all)
#   --sched           LR scheduler: constant, cosine, linear (default: linear)
#   --warmup_ratio    Warmup ratio (default: 0.0)
#   --warmup_steps    Warmup steps, overrides ratio (default: 0)
#   --dtype           Precision: bf16 or fp16 (default: fp16)
#   --no_eval_mmlu    Disable MMLU evaluation
#   --optimizer       Optimizer: adamw or muon (default: adamw)
#   --examples, -n    num_train_examples (default: 0 = full dataset)
#   --seed            Random seed (default: 42)
#   --time            SLURM time limit (default: 6:00:00)
#   --short           Short tamper: 100 steps, eval every 10, default configs
#   --dry-run         Print sbatch script(s) without submitting

set -euo pipefail

# ── defaults ──
MODEL=""
LR=""
STEPS=10000
EVAL_EVERY=500
BS=1
GRAD_ACC=4
EPOCHS=2
DATA="bio_remove"
LORA=0
LORA_TARGET="all"
SCHED=""
WARMUP_RATIO="0.0"
WARMUP_STEPS=0
DTYPE=""
EVAL_MMLU=true
OPTIMIZER="adamw"
EXAMPLES=0
SEED=42
TIME="6:00:00"
SHORT=false
DRY_RUN=false
SAVE_AT_STEP=0
RESUME_FROM=""

# ── parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)       MODEL="$2"; shift 2 ;;
        --lr)             LR="$2"; shift 2 ;;
        --steps)          STEPS="$2"; shift 2 ;;
        --eval_every)     EVAL_EVERY="$2"; shift 2 ;;
        --bs)             BS="$2"; shift 2 ;;
        --grad_acc)       GRAD_ACC="$2"; shift 2 ;;
        --epochs)         EPOCHS="$2"; shift 2 ;;
        --data)           DATA="$2"; shift 2 ;;
        --lora)           LORA="$2"; shift 2 ;;
        --lora_target)    LORA_TARGET="$2"; shift 2 ;;
        --sched)          SCHED="$2"; shift 2 ;;
        --warmup_ratio)   WARMUP_RATIO="$2"; shift 2 ;;
        --warmup_steps)   WARMUP_STEPS="$2"; shift 2 ;;
        --dtype)          DTYPE="$2"; shift 2 ;;
        --eval_mmlu)      EVAL_MMLU=true; shift ;;
        --no_eval_mmlu)   EVAL_MMLU=false; shift ;;
        --optimizer)      OPTIMIZER="$2"; shift 2 ;;
        --examples|-n)    EXAMPLES="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        --short)          SHORT=true; shift ;;
        --save_at_step)   SAVE_AT_STEP="$2"; shift 2 ;;
        --resume)         RESUME_FROM="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$MODEL" ]] && { echo "Error: --model is required"; exit 1; }

if $SHORT; then
    STEPS=100
    EVAL_EVERY=10
    TIME="1:00:00"
fi

REPO_ROOT="/projects/a6a/public/lucia/home/unlearn"

# Extract model tag from path (last path component, strip common prefix)
MODEL_TAG=$(basename "$MODEL")
MODEL_TAG=${MODEL_TAG#deep-ignorance-unfiltered_}

# Resolve model to absolute path if relative
if [[ "$MODEL" != /* ]]; then
    MODEL="${REPO_ROOT}/${MODEL}"
fi

EFF_BATCH=$(( BS * GRAD_ACC * 4 ))
echo "Effective batch size: ${BS} * ${GRAD_ACC} * 4 GPUs = ${EFF_BATCH}"

# ── build config list: each entry is lr:dtype:sched ──
# If user didn't override lr/dtype/sched, use the default 5-config sweep.
# Otherwise, sweep over the specified LRs with the specified dtype/sched.
CONFIGS=()
if [[ -z "$LR" && -z "$DTYPE" && -z "$SCHED" ]]; then
    CONFIGS=(
        "1e-5:fp16:linear"
        "2e-5:fp16:linear"
        "8e-5:fp16:linear"
        "2e-5:bf16:linear"
        "2e-5:fp16:cosine"
    )
else
    [[ -z "$LR" ]] && LR="2e-5"
    [[ -z "$DTYPE" ]] && DTYPE="fp16"
    [[ -z "$SCHED" ]] && SCHED="linear"
    IFS=',' read -ra LR_ARRAY <<< "$LR"
    for lr_val in "${LR_ARRAY[@]}"; do
        CONFIGS+=("${lr_val}:${DTYPE}:${SCHED}")
    done
fi

echo "Submitting ${#CONFIGS[@]} job(s)"

# ── submit one job per config ──
for CFG in "${CONFIGS[@]}"; do

IFS=':' read -r CURRENT_LR CURRENT_DTYPE CURRENT_SCHED <<< "$CFG"

TIMESTAMP=$(date +%m%d_%H%M)
TAG="tamper_${MODEL_TAG}_${DATA}_lr${CURRENT_LR}_s${STEPS}_${CURRENT_SCHED}_${CURRENT_DTYPE}_${TIMESTAMP}"
if [[ "$LORA" -gt 0 ]]; then
    TAG="${TAG}_lora${LORA}_${LORA_TARGET}"
fi
if [[ "$OPTIMIZER" != "adamw" ]]; then
    TAG="${TAG}_${OPTIMIZER}"
fi

OUTPUT_DIR="${REPO_ROOT}/runs/${TAG}"
JOB_NAME="tamp-${MODEL_TAG:0:20}"
OUT_FILE="${REPO_ROOT}/runs/${TAG}-%j.out"

# ── build torchrun command ──
TRAIN_CMD="torchrun --nproc_per_node=4 -m scripts.run_tamper_attack_with_plot \
    --model_name=${MODEL} \
    --output_dir=${OUTPUT_DIR} \
    --lr=${CURRENT_LR} \
    --max_steps=${STEPS} \
    --eval_every=${EVAL_EVERY} \
    --batch_size=${BS} \
    --grad_accumulation=${GRAD_ACC} \
    --epochs=${EPOCHS} \
    --tamper_data=${DATA} \
    --lora_r=${LORA} \
    --lora_target=${LORA_TARGET} \
    --lr_scheduler_type=${CURRENT_SCHED} \
    --warmup_ratio=${WARMUP_RATIO} \
    --warmup_steps=${WARMUP_STEPS} \
    --precision=${CURRENT_DTYPE} \
    --optimizer=${OPTIMIZER} \
    --num_train_examples=${EXAMPLES} \
    --seed=${SEED}"

if $EVAL_MMLU; then
    TRAIN_CMD="$TRAIN_CMD --eval_mmlu"
fi

if [[ "$SAVE_AT_STEP" -gt 0 ]]; then
    TRAIN_CMD="$TRAIN_CMD --save_at_step=${SAVE_AT_STEP}"
fi

if [[ -n "$RESUME_FROM" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint=${RESUME_FROM}"
fi

# ── generate sbatch script ──
SBATCH_SCRIPT=$(cat <<SBEOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=$TIME
#SBATCH --output=$OUT_FILE

echo "============================================"
echo "$TAG"
echo "Job ID: \$SLURM_JOB_ID"
echo "Started: \$(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p \$PIP_CACHE_DIR \$TMPDIR

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available. Exiting."
    exit 1
fi

cd $REPO_ROOT

echo "===== Tamper Attack ====="
echo "Command: $TRAIN_CMD"
$TRAIN_CMD

echo "============================================"
echo "Completed: \$(date)"
echo "============================================"

SBEOF
)

if $DRY_RUN; then
    echo "$SBATCH_SCRIPT"
    echo ""
else
    TMPSCRIPT=$(mktemp /tmp/tamper_XXXXXX.sbatch)
    echo "$SBATCH_SCRIPT" > "$TMPSCRIPT"
    echo "Submitting: $TAG"
    sbatch "$TMPSCRIPT"
    rm "$TMPSCRIPT"
fi

done
