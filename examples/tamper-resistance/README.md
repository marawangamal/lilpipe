# Unfiltered WMDP-Bio LoRA tampering trajectory

This experiment fine-tunes `EleutherAI/deep-ignorance-unfiltered` on the gated
`cais/wmdp-bio-forget-corpus` dataset and evaluates robust WMDP-Bio MCQA
accuracy at optimizer steps 0, 1,000, 2,000, 4,000, 6,000, 8,000, and 10,000.
Step 0 is the untouched Hugging Face model; the remaining points are merged
LoRA checkpoints from one Axolotl run.

## Setup

Python 3.12 and two environments are recommended because Axolotl and lm-eval
have independent dependency stacks:

```bash
uv sync --group train
uv sync --group eval --no-default-groups
```

The Slurm scripts expect those environments at `.venv-train` and `.venv-eval`.
Create them explicitly when running on the cluster:

```bash
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --group train
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --group eval --no-default-groups
```

Request access to
[`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus),
then authenticate the approved account with `hf auth login`. A cached login is
used automatically; an authorized `HF_TOKEN` also works. The preparation script
reports gated-repository errors with an actionable message.

## Run

Inspect the two-job DAG without submitting it:

```bash
lilpipe configs/experiments/unfiltered-wmdp-bio-lora.yml --dry-run
```

Submit by removing `--dry-run`. Training prepares the corpus under
`artifacts/data/`, then lets Axolotl preprocess the prepared text before
training. Both stages exclude `cn-c034`, which currently has a confirmed
filesystem issue reading existing files. Training saves model-only LoRA checkpoints every 1,000
steps and retains all ten. The dependent evaluation stage is a seven-task Slurm array;
the baseline and six selected adapters run concurrently, with one GPU per
array task. After every array task succeeds, generate the table and plot
manually from this directory:

```bash
source .venv-eval/bin/activate
python scripts/analysis/plot_trajectory.py \
  artifacts/evals/unfiltered-wmdp-bio-lora \
  configs/results/unfiltered-wmdp-bio-lora.yml \
  artifacts/results/unfiltered-wmdp-bio-lora
```

This creates:

- `artifacts/results/unfiltered-wmdp-bio-lora/trajectory.csv`
- `artifacts/results/unfiltered-wmdp-bio-lora/trajectory.png`

All paths are resolved from this directory. Generated `artifacts/` are ignored.

For a baseline smoke test (after activating `.venv-eval`):

```bash
python -m lm_eval run --model hf \
  --model_args pretrained=EleutherAI/deep-ignorance-unfiltered,dtype=bfloat16 \
  --include_path lm_eval_tasks --tasks wmdp_bio_robust \
  --num_fewshot 0 --limit 10 \
  --output_path artifacts/evals/smoke-checkpoint-0
```
