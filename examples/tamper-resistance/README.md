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
`artifacts/mila/cache/axolotl/di-6.9b-wmdp-bio-lora-unlearn-cb/prepared`.

## Run

Submit the canonical Circuit Breaker, NPO, and GradDiff methods and their
forget-set relearning stages:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
lilpipe configs/experiments/di-6.9b-lora.yml
```

### DI-6.9B LoRA HPO promotion

The DI LoRA sweeps keep NPO, GradDiff, and Circuit Breaker separate from the
canonical pipeline:

```bash
lilpipe configs/experiments/di-6.9b-lora-hpo-unlearn.yml
lilpipe configs/experiments/di-6.9b-lora-hpo-relearn.yml
```

Each run trains for 250 steps and saves and evaluates every 10 steps. Before
starting relearning, choose the best checkpoint for each method using both
WMDP-Bio and MMLU. Merge that checkpoint's LoRA adapter with
`EleutherAI/deep-ignorance-unfiltered`, place the merged model in a stable
promoted directory, and create the matching link below:

```text
artifacts/mila/models/di-6.9b-wmdp-bio-lora-unlearn-npo-hpo-opt
artifacts/mila/models/di-6.9b-wmdp-bio-lora-unlearn-gd-hpo-opt
artifacts/mila/models/di-6.9b-wmdp-bio-lora-unlearn-cb-hpo-opt
```

Each link must resolve to its method's promoted merged model. The relearning
pipeline trusts these links and starts independent `train.sbatch` jobs; it does
not submit, depend on, or merge an unlearning producer.

DI-6.9B follows the Zephyr LoRA data and optimization setup: the full
WMDP-Bio `train` split uses `wmdp_zephyr`, the Salesforce WikiText-2 `test`
split uses `wikitext2`, merged datasets are not shuffled, and the canonical and
HPO batch, schedule, optimizer, and checkpoint settings match method by method.
Only model-specific settings differ: the base model, GPT-NeoX LoRA target
modules and transformed layers, and artifact identifiers.

### Zephyr-7B LoRA HPO promotion

Unlearning and relearning HPO are separate pipelines:

```bash
lilpipe configs/experiments/z7b-lora-hpo-unlearn.yml
lilpipe configs/experiments/z7b-lora-hpo-relearn.yml
```

Both run NPO, GradDiff, and Circuit Breaker for 250 steps, saving and evaluating
every 10 steps. Promotion between them is manual: choose the best unlearning
checkpoint using WMDP-Bio and MMLU, merge its LoRA adapter with Zephyr into a
stable promoted directory, then create the matching symlink before launching
relearning:

```text
artifacts/mila/models/z7b-wmdp-bio-lora-unlearn-npo-hpo-opt
artifacts/mila/models/z7b-wmdp-bio-lora-unlearn-gd-hpo-opt
artifacts/mila/models/z7b-wmdp-bio-lora-unlearn-cb-hpo-opt
```

Each link must resolve to its method's promoted merged model. Lilpipe does not
create or validate these links.

### Zephyr-7B LoRA canonical configuration

| Stage   | Method   | LR     | Max steps | Checkpoints |
| ------- | -------- | ------ | --------- | ----------- |
| Unlearn | NPO      | `1e-4` | 80        | 8           |
| Unlearn | GradDiff | `2e-4` | 40        | 4           |
| Unlearn | CB       | `5e-4` | 70        | 7           |
| Relearn | NPO      | `3e-5` | 220       | 22          |
| Relearn | GradDiff | `3e-5` | 150       | 15          |
| Relearn | CB       | `1e-4` | 190       | 19          |

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

NPO+SAM (`di-6.9b-wmdp-bio-lora-unlearn-npo-sam`) keeps NPO's β = 0.0225,
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

GradDiff (`di-6.9b-wmdp-bio-lora-unlearn-gd`) uses the same data, balanced batches,
rank-8 LoRA setup, and 32-step training schedule as NPO and CB. It minimizes
`−0.1 × forget CE + 1.0 × retain CE`. Its dependent
relearning model (`di-6.9b-wmdp-bio-lora-unlearn-gd-relearn`) merges the GradDiff
adapter and follows the same 32-step forget-set attack schedule as NPO. Both
models receive the Robust WMDP-Bio and MMLU excluding biology evaluations.

GD-GN (`di-6.9b-wmdp-bio-lora-unlearn-gd-gn`) uses balanced forget and retain
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
one checkpoint's trainable LoRA parameters:

```bash
source "$SCRATCH/lilpipe/examples/tamper-resistance/.venv-train/bin/activate"
python scripts/analysis/per_sample_param_grad_cosim_gen_results.py \
  --model_name_or_path EleutherAI/deep-ignorance-unfiltered \
  --adapter_name_or_path artifacts/mila/models/di-6.9b-wmdp-bio-lora-unlearn-cb/checkpoint-5 \
  --dataset cais/wmdp-bio-forget-corpus \
  --out artifacts/mila/analysis/per_sample_param_grad_cosim_gen_results.json
```

The probe selects 16 documents from the first 1,024 WMDP-Bio training
documents with seed 42 and truncates them to 512 tokens. It reports the mean
cosine over 120 distinct document pairs in one JSON file. The trajectory
plotter remains available for previously generated trajectory JSON files.

Axolotl loads dataset strategies from `configs/training/data/` and objectives
from `configs/training/trainers/`, as selected by `datasets[].type` and
`trainer_cls` in each YAML. DI and Zephyr have separate WMDP strategies because
they format and tokenize documents differently. The NPO, NPO+SAM, GradDiff,
GD-GN, and circuit-breaker trainers share the samplers in
`configs/training/trainers/samplers.py`.
