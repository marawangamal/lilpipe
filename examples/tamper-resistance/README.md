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
`artifacts/cache/axolotl/di-6.9b-wmdp-bio-unlearn-cb/prepared`.

## Run

Submit the base model, circuit breaker and weight-steering methods, and their
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

Axolotl loads the tagged-document strategy and orthogonal trainer from
`configs/unlearn/utils.py`, as selected by `trainer_cls` and
`datasets[].type` in the YAML.
