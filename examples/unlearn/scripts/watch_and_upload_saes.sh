#!/bin/bash
# Watch SAE training jobs and upload to HuggingFace when done

source /home/a6a/lucia.a6a/miniforge3/etc/profile.d/conda.sh
conda activate snake

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.env"
huggingface-cli login --token "$HF_TOKEN"

REPO_ROOT="/home/a6a/lucia.a6a/unlearn"

JOB_EX8=2114668
JOB_EX16=2114687

EX8_DIR="$REPO_ROOT/models/deep-ignorance-unfiltered_saes"
EX16_DIR="$REPO_ROOT/models/deep-ignorance-unfiltered_saes_ex16"

EX8_DONE=0
EX16_DONE=0

echo "Watching jobs $JOB_EX8 (ex8) and $JOB_EX16 (ex16)..."

while [ $EX8_DONE -eq 0 ] || [ $EX16_DONE -eq 0 ]; do
    sleep 60

    if [ $EX8_DONE -eq 0 ]; then
        STATE=$(sacct -j $JOB_EX8 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
        if [ "$STATE" = "COMPLETED" ]; then
            echo "$(date): Job $JOB_EX8 (ex8, all layers) COMPLETED"
            EX8_DONE=1
        elif [ "$STATE" = "FAILED" ] || [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "TIMEOUT" ]; then
            echo "$(date): Job $JOB_EX8 (ex8) ended with state: $STATE"
            EX8_DONE=2
        fi
    fi

    if [ $EX16_DONE -eq 0 ]; then
        STATE=$(sacct -j $JOB_EX16 --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
        if [ "$STATE" = "COMPLETED" ]; then
            echo "$(date): Job $JOB_EX16 (ex16, stride 4) COMPLETED"
            EX16_DONE=1
        elif [ "$STATE" = "FAILED" ] || [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "TIMEOUT" ]; then
            echo "$(date): Job $JOB_EX16 (ex16) ended with state: $STATE"
            EX16_DONE=2
        fi
    fi
done

echo ""
echo "All jobs finished. Uploading to HuggingFace..."

# Upload ex8 SAEs (all 32 layers)
if [ $EX8_DONE -eq 1 ] && [ -d "$EX8_DIR" ]; then
    echo "Uploading ex8 SAEs..."

    cat > "$EX8_DIR/README.md" << 'CARD'
---
library_name: sparsify
base_model: EleutherAI/deep-ignorance-unfiltered
tags:
  - sparse-autoencoder
  - sae
  - mechanistic-interpretability
---

# SAEs for deep-ignorance-unfiltered (expansion factor 8, all layers)

Sparse autoencoders trained on all 32 layers of [EleutherAI/deep-ignorance-unfiltered](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered) using the [WMDP-Bio-Remove-Dataset](https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset).

Trained with [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify).

## Training hyperparameters

| Parameter | Value |
|---|---|
| expansion_factor | 8 |
| k | 32 |
| layers | all 32 (0-31) |
| batch_size | 4 |
| grad_acc_steps | 8 |
| micro_acc_steps | 2 |
| ctx_len | 2048 |
| optimizer | signum |
| loss_fn | fvu |
| activation | topk |
| distributed | distribute_modules across 4x GH200 120GB |

## Dataset

[Unlearning/WMDP-Bio-Remove-Dataset](https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset) (24,453 examples)

## Usage

```python
from sparsify import Sae

sae = Sae.load_from_hub("EleutherAI/sae-deep-ignorance-unfiltered-ex8", hookpoint="layers.10")

# Or load all layers
saes = Sae.load_many("EleutherAI/sae-deep-ignorance-unfiltered-ex8")
```
CARD

    huggingface-cli upload EleutherAI/sae-deep-ignorance-unfiltered-ex8 "$EX8_DIR" . \
        --repo-type model \
        && echo "ex8 upload complete" \
        || echo "ex8 upload FAILED"
fi

# Upload ex16 SAEs (every 4th layer)
if [ $EX16_DONE -eq 1 ] && [ -d "$EX16_DIR" ]; then
    echo "Uploading ex16 SAEs..."

    cat > "$EX16_DIR/README.md" << 'CARD'
---
library_name: sparsify
base_model: EleutherAI/deep-ignorance-unfiltered
tags:
  - sparse-autoencoder
  - sae
  - mechanistic-interpretability
---

# SAEs for deep-ignorance-unfiltered (expansion factor 16, every 4th layer)

Sparse autoencoders trained on every 4th layer (0, 4, 8, 12, 16, 20, 24, 28) of [EleutherAI/deep-ignorance-unfiltered](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered) using the [WMDP-Bio-Remove-Dataset](https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset).

Trained with [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify).

## Training hyperparameters

| Parameter | Value |
|---|---|
| expansion_factor | 16 |
| k | 32 |
| layers | 0, 4, 8, 12, 16, 20, 24, 28 (stride 4) |
| batch_size | 1 |
| grad_acc_steps | 32 |
| micro_acc_steps | 2 |
| ctx_len | 2048 |
| optimizer | signum |
| loss_fn | fvu |
| activation | topk |
| distributed | DDP across 4x GH200 120GB |

## Dataset

[Unlearning/WMDP-Bio-Remove-Dataset](https://huggingface.co/datasets/Unlearning/WMDP-Bio-Remove-Dataset) (24,453 examples)

## Usage

```python
from sparsify import Sae

sae = Sae.load_from_hub("EleutherAI/sae-deep-ignorance-unfiltered-ex16", hookpoint="layers.8")

# Or load all available layers
saes = Sae.load_many("EleutherAI/sae-deep-ignorance-unfiltered-ex16")
```
CARD

    huggingface-cli upload EleutherAI/sae-deep-ignorance-unfiltered-ex16 "$EX16_DIR" . \
        --repo-type model \
        && echo "ex16 upload complete" \
        || echo "ex16 upload FAILED"
fi

echo ""
echo "Done. $(date)"
