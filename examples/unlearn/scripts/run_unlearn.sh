#!/bin/bash
# Universal unlearning job launcher.
#
# Usage:
#   bash scripts/run_unlearn.sh --algorithm checkpoint --rm 5 --ret 5 --rank 16 --lr 1e-4
#   bash scripts/run_unlearn.sh --algorithm lens --rm 5 --ret 0 --rank 32 --examples 1024
#   bash scripts/run_unlearn.sh --algorithm sequential --rm 5 --ret 0 --sft --lr 2e-4
#   bash scripts/run_unlearn.sh --algorithm cb --rm 23 --orth 5 --ret 0 --rank 8
#   bash scripts/run_unlearn.sh --algorithm checkpoint --rm 5 --ret 5 --sft --lr 2e-4
#   bash scripts/run_unlearn.sh --algorithm maxupdate --rm 10 --ret 1 --sft
#
# Options:
#   --algorithm, -a   Algorithm: cb, checkpoint, lens, sequential, maxupdate, module_parallel (required)
#   --rm              remove_coef (required)
#   --ret             retain_coef (required)
#   --rank, -r        LoRA rank (default: 16, ignored with --sft)
#   --lr              Learning rate (default: per-algorithm)
#   --examples, -n    num_train_examples (default: per-algorithm)
#   --pdbs            per-device batch size (default: per-algorithm)
#   --sft             Use full-rank SFT instead of LoRA
#   --orth            orth_coef for circuit breakers (default: 5)
#   --muon            Use Muon optimizer instead of AdamW
#   --nodes           Number of SLURM nodes (default: 1)
#   --time            SLURM time limit (default: 6:00:00)
#   --dtype           Mixed precision: bf16 or fp16 (default: bf16)
#   --tag-suffix      Extra suffix appended to the generated run tag/model path
#   --extra           Extra args passed verbatim to the training script
#   --dry-run         Print the sbatch script without submitting
#   --execute         Execute training in the current Slurm allocation

set -euo pipefail

# ── defaults ──
ALG=""
RM=""
RET=""
RANK=16
LR=""
EXAMPLES=""
PDBS=""
SFT=false
ORTH=5
MUON=false
TIME="6:00:00"
NODES=1
DTYPE="bf16"
TAG_SUFFIX=""
EXTRA=""
DRY_RUN=false
EXECUTE=false
USE_ULTRACHAT=false
WANDB_PROJECT=""

# ── parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm|-a) ALG="$2"; shift 2 ;;
        --rm)           RM="$2"; shift 2 ;;
        --ret)          RET="$2"; shift 2 ;;
        --rank|-r)      RANK="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --examples|-n)  EXAMPLES="$2"; shift 2 ;;
        --pdbs)         PDBS="$2"; shift 2 ;;
        --sft)          SFT=true; shift ;;
        --lora)         SFT=false; shift ;;  # explicit LoRA (default)
        --orth)         ORTH="$2"; shift 2 ;;
        --muon)         MUON=true; shift ;;
        --time)         TIME="$2"; shift 2 ;;
        --nodes)        NODES="$2"; shift 2 ;;
        --dtype)        DTYPE="$2"; shift 2 ;;
        --tag-suffix)   TAG_SUFFIX="$2"; shift 2 ;;
        --extra)        EXTRA="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --execute)      EXECUTE=true; shift ;;
        --use_ultrachat) USE_ULTRACHAT=true; shift ;;
        --wandb)        WANDB_PROJECT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$ALG" ]] && { echo "Error: --algorithm is required (cb, checkpoint, lens, sequential, maxupdate, module_parallel)"; exit 1; }
[[ -z "$RM" ]]  && { echo "Error: --rm is required"; exit 1; }
[[ -z "$RET" ]] && { echo "Error: --ret is required"; exit 1; }



# ── per-algorithm defaults ──
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LENS_DIR="$REPO_ROOT/runs/lens_cache/deep-ignorance-unfiltered-lens"
TRAIN_CMD=""
EVAL_MODEL=""

case "$ALG" in
    cb|circuit_breakers)
        [[ -z "$LR" ]] && LR="1e-3"
        [[ -z "$EXAMPLES" ]] && EXAMPLES=1024
        [[ -z "$PDBS" ]] && PDBS=1
        if $SFT; then
            TAG="cb_sft_ret${RET}_rm${RM}_orth${ORTH}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="python -m unlearn.algorithm.orth_circuit_breakers \
    --remove_coef=$RM --retain_coef=$RET --orth_coef=$ORTH \
    --lora=False --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --retain_warmup=True \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0"
            TRAIN_GPU_COUNT=1
        else
            TAG="cb_lora_ret${RET}_rm${RM}_orth${ORTH}_r${RANK}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="python -m unlearn.algorithm.orth_circuit_breakers \
    --remove_coef=$RM --retain_coef=$RET --orth_coef=$ORTH \
    --lora_r=$RANK --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --retain_warmup=True \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0"
            TRAIN_GPU_COUNT=1
        fi
        ;;

    checkpoint|ct)
        [[ -z "$LR" ]] && LR="1e-3"
        [[ -z "$EXAMPLES" ]] && EXAMPLES=2048
        [[ -z "$PDBS" ]] && PDBS=2
        if $SFT; then
            TAG="ct_sft_ret${RET}_rm${RM}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.checkpoint_transfer_unlearn \
    --remove_coef=$RM --retain_coef=$RET \
    --lora=False --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --retain_ce_loss=True --retain_kl_loss=False \
    --load_affine_from_hub=EleutherAI/affine-checkpoint-transfer \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0,1,2,3"
        else
            TAG="ct_lora_ret${RET}_rm${RM}_r${RANK}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.checkpoint_transfer_unlearn \
    --remove_coef=$RM --retain_coef=$RET \
    --lora=True --lora_r=$RANK --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --load_affine_from_hub=EleutherAI/affine-checkpoint-transfer \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH/merged"
            TRAIN_GPUS="0,1,2,3"
        fi
        ;;

    lens)
        [[ -z "$LR" ]] && LR="1e-3"
        [[ -z "$EXAMPLES" ]] && EXAMPLES=1024
        [[ -z "$PDBS" ]] && PDBS=1
        OPTIMIZER_FLAG="adamw"
        $MUON && OPTIMIZER_FLAG="muon"
        if $SFT; then
            TAG="lens_sft_ret${RET}_rm${RM}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.lens_unlearn \
    --remove_coef=$RM --retain_coef=$RET \
    --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --optimizer=$OPTIMIZER_FLAG \
    --lens_path=$LENS_DIR \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0,1,2,3"
        else
            TAG="lens_lora_ret${RET}_rm${RM}_r${RANK}_lr${LR}_ex${EXAMPLES}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.lens_unlearn \
    --remove_coef=$RM --retain_coef=$RET \
    --lora_r=$RANK --lora=True --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --optimizer=$OPTIMIZER_FLAG \
    --lens_path=$LENS_DIR \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH/merged"
            TRAIN_GPUS="0,1,2,3"
        fi
        ;;

    sequential|seq)
        [[ -z "$LR" ]] && LR="2e-4"
        [[ -z "$EXAMPLES" ]] && EXAMPLES=1024
        [[ -z "$PDBS" ]] && PDBS=1
        if $SFT; then
            TAG="seq_sft_ret${RET}_rm${RM}_lr${LR}_nn${NODES}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.sequential_retain_sft \
    --remove_coef=$RM --retain_coef=$RET \
    --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --start_layer=31 --end_layer=0 --layer_step=1 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0,1,2,3"
        else
            TAG="seq_lora_ret${RET}_rm${RM}_r${RANK}_lr${LR}_nn${NODES}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="python -m unlearn.algorithm.sequential_retain \
    --remove_coef=$RM --retain_coef=$RET \
    --lora_r=$RANK --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --start_layer=31 --end_layer=0 --layer_step=1 \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
        fi
        ;;

    module_parallel|mp)
        [[ -z "$LR" ]] && LR="1e-4"
        [[ -z "$EXAMPLES" ]] && EXAMPLES=1024
        [[ -z "$PDBS" ]] && PDBS=1
        TAG="mp_sft_ret${RET}_rm${RM}_lr${LR}"
        MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
        TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.module_parallel_unlearn \
    --remove_coef=$RM --retain_coef=$RET \
    --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
        EVAL_MODEL="$MODEL_PATH"
        TRAIN_GPUS="0,1,2,3"
        ;;

    maxupdate|mu)
        [[ -z "$LR" ]] && LR="2e-4"
        [[ -z "$EXAMPLES" ]] && EXAMPLES=1024
        [[ -z "$PDBS" ]] && PDBS=1
        if $SFT; then
            TAG="mu_sft_ret${RET}_up${RM}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.max_update_unlearn \
    --update_coef=$RM --retain_coef=$RET \
    --lora=False --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0,1,2,3"
        else
            TAG="mu_lora_ret${RET}_up${RM}_r${RANK}_lr${LR}"
            MODEL_PATH="$REPO_ROOT/models/EleutherAI/deep-ignorance-unfiltered_${TAG}"
            TRAIN_CMD="torchrun --nproc_per_node=4 -m unlearn.algorithm.max_update_unlearn \
    --update_coef=$RM --retain_coef=$RET \
    --lora=True --lora_r=$RANK --lr=$LR --pdbs=$PDBS --num_train_examples=$EXAMPLES \
    --model_name=EleutherAI/deep-ignorance-unfiltered \
    --save_path=$MODEL_PATH $EXTRA"
            EVAL_MODEL="$MODEL_PATH"
            TRAIN_GPUS="0,1,2,3"
        fi
        ;;

    *)
        echo "Unknown algorithm: $ALG (use cb, checkpoint, lens, sequential, maxupdate, module_parallel)"
        exit 1
        ;;
esac

# ── default TRAIN_GPUS if not set by algorithm ──
: "${TRAIN_GPUS:=0}"
: "${TRAIN_GPU_COUNT:=4}"

# ── dtype handling ──
if [[ "$DTYPE" != "bf16" ]]; then
    TAG="${TAG}_${DTYPE}"
fi
TRAIN_CMD="$TRAIN_CMD --dtype=$DTYPE"

# ── muon handling ──
if $MUON; then
    TAG="${TAG}_muon"
    # lens uses a separate module; other algorithms use --optimizer=muon
    if [[ "$ALG" != "lens" ]]; then
        TRAIN_CMD="$TRAIN_CMD --optimizer=muon"
    fi
fi

# ── ultrachat handling ──
if $USE_ULTRACHAT; then
    TAG="${TAG}_ultrachat"
    TRAIN_CMD="$TRAIN_CMD --use_ultrachat=True"
fi

if [[ -n "$TAG_SUFFIX" ]]; then
    TAG="${TAG}_${TAG_SUFFIX}"
fi

# ── wandb handling (after TAG is finalized) ──
if [[ -n "$WANDB_PROJECT" ]]; then
    TRAIN_CMD="$TRAIN_CMD --wandb_project=$WANDB_PROJECT --wandb_run_name=$TAG"
fi

# ── sweep over LRs (comma-separated) ──
IFS=',' read -ra LR_ARRAY <<< "$LR"

for SWEEP_LR in "${LR_ARRAY[@]}"; do

CUR_TAG=$(echo "$TAG" | sed "s|lr${LR}|lr${SWEEP_LR}|")
CUR_MODEL_PATH="$REPO_ROOT/artifacts/models/$CUR_TAG"
if [[ "$EVAL_MODEL" == *"/merged" ]]; then
    CUR_EVAL_MODEL="$CUR_MODEL_PATH/merged"
else
    CUR_EVAL_MODEL="$CUR_MODEL_PATH"
fi
CUR_TRAIN_CMD=$(echo "$TRAIN_CMD" | sed "s|--lr=${LR}|--lr=${SWEEP_LR}|")
CUR_TRAIN_CMD=$(echo "$CUR_TRAIN_CMD" | sed "s|--save_path=[^ ]*|--save_path=$CUR_MODEL_PATH|")

if $EXECUTE; then
    echo "Executing: $CUR_TRAIN_CMD"
    eval "$CUR_TRAIN_CMD"
    continue
fi

# ── multi-node handling ──
if [[ "$NODES" -gt 1 ]]; then
    if [[ "$CUR_TRAIN_CMD" != torchrun* ]]; then
        echo "Error: --nodes > 1 only supported for torchrun-based algorithms"
        exit 1
    fi
    TRAIN_SUFFIX="${CUR_TRAIN_CMD#torchrun --nproc_per_node=4 }"
    TRAIN_BLOCK='MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
MASTER_PORT=29500
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

srun --ntasks-per-node=1 torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    '"$TRAIN_SUFFIX"
else
    TRAIN_BLOCK="$CUR_TRAIN_CMD"
fi

JOB_NAME="unlearn-${CUR_TAG}"
OUT_FILE="$REPO_ROOT/runs/${CUR_TAG}-%j.out"

# ── generate sbatch script ──
SBATCH_SCRIPT=$(cat <<SBEOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NODES
#SBATCH --exclusive
#SBATCH --gpus-per-node=$TRAIN_GPU_COUNT
#SBATCH --ntasks-per-node=1
#SBATCH --time=$TIME
#SBATCH --output=$OUT_FILE

echo "============================================"
echo "$CUR_TAG"
echo "Job ID: \$SLURM_JOB_ID"
echo "Started: \$(date)"
echo "============================================"

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

module load PrgEnv-cray 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
module load brics/nccl/2.21.5-1 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONUNBUFFERED=1
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export PIP_CACHE_DIR="/projects/a6a/public/lucia/pip_cache"
export TMPDIR="/projects/a6a/public/lucia/tmp"
if [[ -n "$WANDB_PROJECT" ]]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=disabled
fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p \$PIP_CACHE_DIR \$TMPDIR

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available. Exiting."
    exit 1
fi

cd $REPO_ROOT

echo "===== Training ====="
$TRAIN_BLOCK

if [ ! -d "$CUR_EVAL_MODEL" ]; then
    echo "ERROR: Model not found at $CUR_EVAL_MODEL"
    exit 1
fi

if [ ! -f "$CUR_EVAL_MODEL/config.json" ]; then
    echo "ERROR: Model config missing at $CUR_EVAL_MODEL/config.json"
    exit 1
fi

echo "===== WMDP + MMLU Eval ====="

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HUB_OFFLINE=1

OUTPUT_JSON="$REPO_ROOT/runs/evals/eval_results_$CUR_TAG.json"
TASKS="wmdp_bio_robust,mmlu"

JOB_ID=\$(sbatch --parsable \
    "$REPO_ROOT/scripts/eval_checkpoint.sbatch" \
    "$CUR_EVAL_MODEL" \
    "\$OUTPUT_JSON" \
    "\$TASKS")

echo "Eval job \$JOB_ID submitted. Writing results to \$OUTPUT_JSON"


SBEOF
)

if $DRY_RUN; then
    echo "$SBATCH_SCRIPT"
else
    TMPSCRIPT=$(mktemp /tmp/unlearn_XXXXXX.sbatch)
    echo "$SBATCH_SCRIPT" > "$TMPSCRIPT"
    echo "Submitting: $CUR_TAG"
    sbatch "$TMPSCRIPT"
    rm "$TMPSCRIPT"
fi

done
