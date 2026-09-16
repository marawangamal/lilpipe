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

## Run

Submit the isolated 25-step smoke run, wait for it to finish, and enforce the
rerouting safeguard before submitting the canonical rank-16 CB run:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/unfiltered-cb--repr-smoke.yml
# Run only after the smoke Slurm job has completed.
python scripts/training/check_cb_smoke.py artifacts/smoke/models/unfiltered-cb--repr
lilpipe configs/experiments/unfiltered-cb--repr.yml
```

The checker requires nonzero learned LoRA-B weights and at least a 0.001 drop
in harmful activation cosine. Once canonical training finishes, evaluate its
step-150 adapter with the shared evaluator (alongside separately recorded base
and released-CB baselines):

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
