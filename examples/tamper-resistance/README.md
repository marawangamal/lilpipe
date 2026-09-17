# Deep Ignorance circuit-breaker reproduction and tampering

Fine-tune `DI-6.9B-Base` (`EleutherAI/deep-ignorance-unfiltered`) on WMDP-Bio and evaluate
robust MCQA accuracy across checkpoints. Run every command below from this
directory (`examples/tamper-resistance`).

The bundled CB configuration is a behavioral reproduction: the paper does not
publish every original training hyperparameter, so it uses one fixed,
paper-informed GraySwan-style configuration rather than claiming an exact
checkpoint recreation.

## Setup

Request access to
[`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
and log in with `hf auth login`.

```bash
# add env vars
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export HF_HOME="$SCRATCH/.cache/huggingface"

# keep generated data, checkpoints, and results on scratch
mkdir -p "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts"
ln -sfnT "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts" artifacts

# install environments
UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train" \
uv sync --group train
UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval" \
uv sync --group eval
```

Slurm jobs install clean environments and uv caches under `$SLURM_TMPDIR`.
The persistent environments above are only for submitting pipelines and plotting.
Axolotl's reusable tokenized circuit-breaker dataset is stored under
`artifacts/cache/axolotl/di-6.9b-cb`.

## Run

Submit the single experiment containing the base model, circuit breaker, and
weight-steering alpha sweep:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/di-6.9b.yml
```

The pipeline trains the retained rank-64 CB and LAT adapters, constructs the
four steered adapters, and evaluates all six models on Robust WMDP-Bio and
MMLU excluding biology. Render the corresponding three-group results table with:

```bash
lilpipe results configs/results/di-6.9b.yml
```

Axolotl loads both the custom trainer and paired-data strategy from
`configs/training/utils.py`, as selected by `trainer_cls` and
`datasets[].type` in the YAML.

## LAT weight-steering unlearning

The weight-steering experiment trains matched rank-64 LoRA adapters for one epoch on the
harmless (`chosen`) and harmful (`rejected`) responses in
`LLM-LAT/harmful-dataset`, then constructs

\[
\theta_{\mathrm{ws}}(\alpha) = \theta_0 +
\alpha(\Delta\theta_{\mathrm{chosen}}-\Delta\theta_{\mathrm{rejected}}).
\]

Thus the chosen adapter has weight `alpha` and the rejected adapter has weight
`-alpha`. Alpha 1 applies the learned contrast once; alpha 3, alpha 5, and alpha 10
extrapolate farther in the same harmless-minus-harmful direction. Submit the complete DAG
from this directory:

```bash
lilpipe configs/experiments/di-6.9b.yml
```

The selected steering models pull both training arms into the plan through
their dependencies:

```text
DI-6.9B-Base
├─> di-6.9b-ft-lat-chosen ──┐
└─> di-6.9b-ft-lat-rejected ┴─> di-6.9b-w-steer-lat-reject2accept-a-{1,3,5,10} ─> {bio-mcqa,mmlu-no-bio}
```

The final adapters are written below `artifacts/models/<model-id>`. Robust
WMDP-Bio zero-shot results for the steered adapters are written below
`artifacts/evals/<model-id>/wmdp-bio-robust/`.

## Naming migration provenance

Repository IDs and paths use lowercase canonical slugs. Documentation and W&B
use display case; upstream Hugging Face identifiers remain unchanged.

| Previous ID | Canonical ID |
|---|---|
| `unfiltered-ft-lat-chosen` | `di-6.9b-lora16-steps150-lat-chosen` |
| `unfiltered-ft-lat-rejected` | `di-6.9b-lora16-steps150-lat-rejected` |
| `unfiltered-ws-a-{1,2,3,5,10}` | `di-6.9b-lora16-steps150-w-steer-a-{1,2,3,5,10}` |
| `unfiltered-ft-lat-chosen-cb-matched` | `di-6.9b-ft-lat-chosen` |
| `unfiltered-ft-lat-rejected-cb-matched` | `di-6.9b-ft-lat-rejected` |
| `unfiltered-ws-cb-matched-a-{1,2,3,5,10}` | `di-6.9b-w-steer-lat-reject2accept-a-{1,2,3,5,10}` |
| `unfiltered-cb--repr` | `di-6.9b-cb` |
| `unfiltered-wmdp-bio-lora` | `di-6.9b-lora16-steps2000-wmdp-bio` |
| `weak-filter-wmdp-bio-lora` | `di-6.9b-weak-filter-lora16-steps2000-wmdp-bio` |
| `unfiltered-cb-wmdp-bio-lora` | `di-6.9b-cb-lora16-steps2000-wmdp-bio` |
| `unfiltered-cb-wmdp-bio-lora--repr` | `di-6.9b-cb--lora16-steps2000-wmdp-bio` |

The shared circuit-breaker model directory is the recoverable artifact for W&B
run `7aeae0rg`. The other eight archived runs retain their W&B configurations
and histories but have no recoverable model weights: every run wrote to the
same local output directory, later runs overwrote it, and W&B has no logged
model-weight artifacts for them.
