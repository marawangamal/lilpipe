# Deep Ignorance circuit-breaker reproduction and tampering

Fine-tune `EleutherAI/deep-ignorance-unfiltered` on WMDP-Bio and evaluate
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
`artifacts/cache/axolotl/circuit-breaker`.

## Run

Submit the canonical rank-16 CB run:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/unfiltered-cb--repr.yml
```

Once canonical training finishes, evaluate its step-150 adapter with the shared
evaluator (alongside separately recorded base and released-CB baselines):

```bash
sbatch --array=1 scripts/slurm/eval_wmdp_bio_mcqa.sbatch \
  artifacts unfiltered-cb--repr EleutherAI/deep-ignorance-unfiltered 150
```

Submit the 2,000-step attack only if the corrected checkpoint's Robust MCQA
accuracy is below the base model's. The attack runs against the merged local
model and evaluates its attack checkpoints on Robust MCQA:

```bash
lilpipe configs/experiments/unfiltered-cb-wmdp-bio-lora--repr.yml
```

Axolotl loads both the custom trainer and paired-data strategy from
`configs/training/utils.py`, as selected by `trainer_cls` and
`datasets[].type` in the YAML.

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/unfiltered-wmdp-bio-lora.yml
```

After the training and evaluation jobs finish:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval/bin/activate"
python scripts/analysis/plot_trajectory.py \
  artifacts/evals/unfiltered-wmdp-bio-lora \
  artifacts/results/unfiltered-wmdp-bio-lora
```

Training writes PEFT adapters at steps 50, 100, and 150, a final adapter in
`artifacts/models/unfiltered-cb--repr`, and merged weights in its `merged/`
subdirectory. The attack writes checkpoints every 250 steps. The shared
`eval_wmdp_bio_mcqa.sbatch` array evaluator receives the base model explicitly
and evaluates each corresponding attack adapter from step 250 through step
2,000. Evaluation and plots live under `artifacts/evals/` and
`artifacts/results/` respectively.

## LAT weight-steering unlearning

The weight-steering experiment trains matched rank-16 LoRA adapters on the
harmless (`chosen`) and harmful (`rejected`) responses in
`LLM-LAT/harmful-dataset`, then constructs

\[
\theta_{\mathrm{ws}}(\alpha) = \theta_0 +
\alpha(\Delta\theta_{\mathrm{chosen}}-\Delta\theta_{\mathrm{rejected}}).
\]

Thus the chosen adapter has weight `alpha` and the rejected adapter has weight
`-alpha`. Alpha 1 applies the learned contrast once; alpha 2, alpha 5, and alpha 10
extrapolate farther in the same harmless-minus-harmful direction. Submit the complete DAG
from this directory:

```bash
lilpipe configs/experiments/unfiltered-weight-steering.yml
```

The selected steering models pull both training arms into the plan through
their dependencies:

```text
deep-ignorance-unfiltered
├─> unfiltered-ft-lat-chosen ──┐
└─> unfiltered-ft-lat-rejected ┴─> unfiltered-ws-a-{1,2,5,10} ─> bio-mcqa
```

The four final adapters are written below `artifacts/models/<model-id>`. Robust
WMDP-Bio zero-shot results for the steered adapters are written below
`artifacts/evals/<model-id>/wmdp-bio-robust/`.
