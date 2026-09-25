# Deep Ignorance unlearning and tampering

Fine-tune `DI-6.9B-Base` (`EleutherAI/deep-ignorance-unfiltered`) on WMDP-Bio and evaluate
robust MCQA accuracy across checkpoints. Run every command below from this
directory (`examples/tamper-resistance`).

The bundled CB configuration is a behavioral reproduction of the
`examples/unlearn` circuit breaker run, not an identical implementation.

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
Axolotl's tokenized WMDP/WikiText dataset is stored under
`artifacts/mila/cache/axolotl/di-6.9b-wmdp-bio-unlearn-cb/prepared`.

## Run

Submit the base model, circuit breaker, NPO, NPO+SAM, GradDiff, GD-GN, and weight-steering methods, and their
forget-set relearning stages:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/di-6.9b.yml
```

The pipeline trains the rank-8 circuit breaker on 1,024 WMDP-Bio forget
documents and 1,024 WikiText retain documents, with four examples from each
source per eight-sample microbatch and eight accumulation steps. Its coefficients
ramp from retain 1 to 10, reroute 23 to 17.25, and orthogonalization 0 to 5; the
learning rate starts at `1e-3` and decays linearly without warmup. It evaluates
the base and trained models on Robust WMDP-Bio and MMLU excluding biology.
The relearning stage merges the CB adapter into the base model, then fine-tunes a
new rank-8 adapter on the 1,024 WMDP-Bio forget documents for 32 steps at
`1e-3` using a 2 × 16 batch/accumulation schedule.

NPO uses the same documents, rank-8 LoRA modules, balanced eight-row
microbatches, 32-step budget, and optimizer schedule as CB. Its loss uses the
forget-set cross-entropy with the adapter enabled and the frozen base-model
cross-entropy with the adapter disabled: `−(2/β) log σ(β × (current CE −
reference CE)) + γ × retain CE`, with β = 0.0225 and γ = 1.0. These constants
are defined in `configs/training/trainers/npo.py`, because Axolotl drops unknown YAML
fields. The dependent NPO relearning stage merges its adapter and uses the same
32-step forget-set attack as CB. Both NPO models are evaluated on Robust
WMDP-Bio and MMLU excluding biology.

NPO+SAM (`di-6.9b-wmdp-bio-unlearn-npo-sam`) keeps NPO's β = 0.0225,
retain weight γ = 1.0, data, LoRA setup, and 32-step schedule. For each
microbatch it maximizes the forget loss within a radius ρ = 0.01 in trainable
LoRA weight space, then accumulates the perturbed forget gradient and the
unperturbed WikiText retain gradient. Its dependent relearning stage uses the
same 32-step forget-set attack. This is a controlled LoRA comparison, rather
than the [paper's](https://arxiv.org/pdf/2502.05374) full-model NPO+SAM tuning.
Both stages receive Robust WMDP-Bio and MMLU excluding biology evaluations.

The NPO+SAM tuning sweep keeps the data, LoRA setup, optimizer, and 32-step
schedule fixed. It compares a smaller LoRA perturbation (`rho003`: ρ = 0.003),
stronger retention (`gamma225`: γ = 2.25), and the released NPO+SAM β/γ pair
(`beta015-gamma225`: β = 0.015, γ = 2.25). The latter still trains only LoRA
weights, so it is not a reproduction of the paper's full-model run. Each
variant uses the same 32-step forget-set relearning attack and both evaluation
tasks before and after relearning.

A follow-up run doubles the retain weight from γ = 2.25 to γ = 4.5
(`gamma450`) while keeping β = 0.0225 and ρ = 0.01. It uses the same data,
LoRA setup, 32-step unlearning and relearning schedules, and evaluations.
The `gamma900` follow-up doubles the retain weight again to γ = 9.0, with
the remaining settings fixed.

GradDiff (`di-6.9b-wmdp-bio-unlearn-gd`) uses the same data, balanced batches,
rank-8 LoRA setup, and 32-step training schedule as NPO and CB. It minimizes
`−0.1 × forget CE + 1.0 × retain CE`. Its dependent
relearning model (`di-6.9b-wmdp-bio-unlearn-gd-relearn`) merges the GradDiff
adapter and follows the same 32-step forget-set attack schedule as NPO. Both
models receive the Robust WMDP-Bio and MMLU excluding biology evaluations.

GD-GN (`di-6.9b-wmdp-bio-unlearn-gd-gn`) uses balanced forget and retain
batches and the same rank-8 LoRA setup. Its objective is
`−forget CE + 0.01 × ||∇LoRA forget CE||₂ + retain CE`. The gradient norm stays
in the autograd graph so training includes its second-order derivative. Its
microbatch is two rows with 32 accumulation steps, preserving the other methods'
effective batch size of 64. The config requests an 80 GB A100 and eager
attention because the flash-attention backward kernel has no second derivative.
The full-context run still exceeded A100 memory on its first gradient
calculation, so no successful run is available. The configured relearning stage
uses the same 32-step forget-set attack and both stages have the same evaluations.

Weight steering trains separate rank-8 LoRA adapters on the same 1,024 WikiText
retain and WMDP-Bio forget documents, using the CB optimizer, learning rate,
32-step schedule, and batch settings. It builds a retain-minus-forget adapter
with coefficient 1, then merges that adapter and applies the same relearning
configuration as CB.

The α=2, 4, and 10 weight-steering variants reuse the same trained retain and
forget arms. Their separate adapters are evaluated on the same two tasks.

Render the results table with:

```bash
lilpipe results configs/results/di-6.9b.yml
```

Measure agreement between per-document forget-loss gradients with respect to
each checkpoint's effective dense LoRA update, then plot CB and NPO across
unlearning and relearning:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
python scripts/analysis/per_sample_param_grad_cosim_gen_results.py
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval/bin/activate"
python scripts/analysis/per_sample_param_grad_cosim_plot_results.py
```

The probe selects 16 documents once from the first 1,024 WMDP-Bio training
documents with seed 42 and truncates them to 512 tokens. It reports the mean
cosine over 120 distinct document pairs at steps 5, 10, 15, 20, 25, 30, and
32 in each stage. Relearning steps are plotted with a 32-step offset. The
outputs are `artifacts/mila/analysis/per_sample_param_grad_cosim.json` and `.png`.
The generator needs the CB and NPO checkpoints and their merged models. Its
temporary factor files require free disk space and are removed after each
checkpoint.

Axolotl loads dataset strategies from `configs/training/data/` and objectives
from `configs/training/trainers/`, as selected by `datasets[].type` and
`trainer_cls` in each YAML. DI and Zephyr have separate WMDP strategies because
they format and tokenize documents differently. The NPO, NPO+SAM, GradDiff,
GD-GN, and circuit-breaker trainers share the samplers in
`configs/training/trainers/samplers.py`.
