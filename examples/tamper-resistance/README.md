# Deep Ignorance unlearning and tampering

Fine-tune `DI-6.9B-Base` (`EleutherAI/deep-ignorance-unfiltered`) on WMDP-Bio and evaluate
robust MCQA accuracy across checkpoints. Run every command below from this
directory (`examples/tamper-resistance`).

The bundled CB configuration is a behavioral reproduction of the
`examples/unlearn` circuit breaker run, not an identical implementation.

The `z7b-cb-tar.yml` pipeline treats unlearning as a composable producer:
merged Zephyr Circuit Breaker → rank-8 CB-TAR → forget-set relearning. This
compute-reduced implementation samples one final eight-step SFT attack endpoint
and differentiates TAR's first-order objective through LoRA parameters. It is
not a full-parameter, multi-coordinate reproduction of the TAR paper.

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
starting relearning, select checkpoints using the same rule as the Zephyr HPO:

1. Keep checkpoints with WMDP-Bio Robust MCQA accuracy at or below 28%, close
   to the 25% random-choice floor.
2. For each method, promote the qualifying checkpoint with the highest
   MMLU-no-bio accuracy.

Merge that checkpoint's LoRA adapter with
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
every 10 steps. Promotion between them is manual: retain checkpoints at or
below 28% WMDP-Bio Robust MCQA, then select the highest MMLU-no-bio checkpoint
for each method. Merge its LoRA adapter with Zephyr directly into the matching
stable promoted directory before launching relearning:

```text
artifacts/mila/models/z7b-wmdp-bio-lora-unlearn-npo-hpo-opt
artifacts/mila/models/z7b-wmdp-bio-lora-unlearn-gd-hpo-opt
artifacts/mila/models/z7b-wmdp-bio-lora-unlearn-cb-hpo-opt
```

Each path must contain its method's promoted merged model. Lilpipe does not
create or validate these directories.

### Zephyr-7B matched 125-step GD and GD+SAM

The matched LoRA study runs 125 unlearning steps followed by 125 relearning
steps. Sweep configs live directly in `configs/unlearn/z7b/hpo` and
`configs/relearn/z7b/hpo`; their filenames carry the `lora-s125` identity.
Run the unlearning sweep, relearning sweep, or selected canonical pipeline with:

```bash
lilpipe configs/experiments/z7b-lora-s125-hpo-unlearn-gd-sam.yml
lilpipe configs/experiments/z7b-lora-s125-hpo-relearn-gd-sam.yml
lilpipe configs/experiments/z7b-lora-s125.yml
```

The canonical unlearning runs use `7.5e-5`; GD+SAM additionally uses
`rho=1e-3`. Their checkpoint-125 WMDP-Bio/MMLU-no-bio scores are 23.96%/44.38%
for GD and 23.85%/40.71% for GD+SAM. Both canonical relearning runs use
`1e-4`, reaching 55.88%/58.44% from GD and 56.57%/58.29% from GD+SAM.

### Zephyr-7B LoRA canonical configuration

| Stage   | Method   | LR     | Max steps | Checkpoints |
| ------- | -------- | ------ | --------- | ----------- |
| Unlearn | NPO      | `1e-4` | 80        | 8           |
| Unlearn | GradDiff | `2e-4` | 40        | 4           |
| Unlearn | CB       | `5e-4` | 70        | 7           |
| Unlearn | GD-DEFF  | `5e-5` | 140*      | 14          |
| Unlearn | GD+SAM   | `5e-5` | 170*      | 17          |
| Relearn | NPO      | `3e-5` | 220       | 22          |
| Relearn | GradDiff | `3e-5` | 150       | 15          |
| Relearn | CB       | `1e-4` | 250       | 25          |
| Relearn | GD-DEFF  | `1e-4` | 250       | 25          |

The canonical CB trajectory is copied from the selected HPO runs: unlearning
uses `5e-4` through step 70, while relearning uses `1e-4` through step 250.
The selected unlearning checkpoint at step 70 scores 27.30% WMDP-Bio Robust
MCQA and 50.15% MMLU-no-bio. The relearning sweep used that checkpoint after
merging its LoRA adapter into Zephyr; checkpoint 190 was the selected attack
point, scoring 55.07% WMDP-Bio and 57.13% MMLU-no-bio, while the canonical
artifact retains the complete trajectory through step 250.

The canonical GD-DEFF artifact promotes checkpoint 140 from its 250-step HPO
run, preserving the original linear schedule with 12 warmup steps. That
checkpoint scores 27.88% WMDP-Bio Robust MCQA and 54.76% MMLU-no-bio. The
canonical artifact retains checkpoints 10 through 140 and merges checkpoint
140; it does not retrain with a shortened 140-step schedule, which would change
the linear learning-rate trajectory. Relearning uses the full `1e-4` HPO run;
checkpoint 50 is the selected attack point, scoring 56.91% WMDP-Bio and 58.90%
MMLU-no-bio, while the canonical artifact retains all checkpoints through step
250.

The canonical GD+SAM artifact likewise promotes checkpoint 170 from its
250-step HPO run at `rho=0.01`, preserving the original linear schedule. That
checkpoint scores 28.00% WMDP-Bio Robust MCQA and 50.17% MMLU-no-bio. Its
relearning sweep uses learning rates `1e-6`, `3e-6`, `1e-5`, `3e-5`, and
`1e-4` for 250 steps, evaluating every 10 steps.

CB trains rank-8 adapters on 1,024 WMDP-Bio forget documents and 1,024
WikiText retain documents. Unlearning uses a 4 × 2 microbatch/accumulation
schedule; its coefficients ramp from retain 1 to 10, reroute 23 to 17.25, and
orthogonalization 0 to 5. Relearning fine-tunes a new rank-8 adapter on the
forget documents with a 1 × 4 schedule. Both stages use a linear scheduler with
12 warmup steps and are evaluated on Robust WMDP-Bio and MMLU excluding
biology. Both stages save every 10 steps.

NPO uses the same documents, rank-8 LoRA modules, balanced eight-row
microbatches, 32-step budget, and optimizer schedule as CB. Its loss uses the
forget-set cross-entropy with the adapter enabled and the frozen base-model
cross-entropy with the adapter disabled: `−(2/β) log σ(β × (current CE −
reference CE)) + γ × retain CE`, with β = 0.0225 and γ = 1.0. These constants
are defined in `configs/training/trainers/npo.py`, because Axolotl drops unknown YAML
fields. The dependent NPO relearning stage merges its adapter and uses the same
32-step forget-set attack as CB. Both NPO models are evaluated on Robust
WMDP-Bio and MMLU excluding biology.


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
