# OpenUnlearning WMDP pipeline

This directory vendors
[`locuslab/open-unlearning`](https://github.com/locuslab/open-unlearning) at commit
`4ad738aaf60f6a4385f6e2506d01da99e76c31f3` and adds a lilpipe pipeline for
the repository's WMDP-Cyber RMU experiment. Run all commands from this directory.

The pipeline compares the original `HuggingFaceH4/zephyr-7b-beta` model with the
RMU-unlearned model. Training uses `cyber-forget-corpus.jsonl` and
`cyber-retain-corpus.jsonl`; both models are then evaluated with lm-eval on
`wmdp_cyber` and MMLU.

## Setup

OpenUnlearning pins Python packages tightly. Populate the shared uv cache once from a
login node; each Slurm job creates a clean environment under `$SLURM_TMPDIR` from this
cache:

```bash
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[lm-eval]'
```

The lilpipe jobs override the upstream FlashAttention setting with PyTorch's
portable eager attention implementation, so a CUDA compiler is not required
during setup.

Download the WMDP text corpora used for unlearning:

```bash
.venv/bin/python setup_data.py --wmdp
```

For cluster storage, place generated artifacts on scratch:

```bash
mkdir -p "$SCRATCH/lilpipe/examples/open-unlearning/artifacts"
ln -sfnT "$SCRATCH/lilpipe/examples/open-unlearning/artifacts" artifacts
export HF_HOME="$SCRATCH/.cache/huggingface"
```

## Run

Inspect the three-job DAG (one unlearning job and two evaluations):

```bash
lilpipe configs/lilpipe/experiments/wmdp-cyber-rmu.yml --dry-run
```

Submit it:

```bash
lilpipe configs/lilpipe/experiments/wmdp-cyber-rmu.yml
```

The base-model evaluation can start immediately. The unlearned-model evaluation
depends on the RMU job. Model weights and evaluation JSON files are written under
`artifacts/models/` and `artifacts/evals/`, respectively.

The Cyber manifest evaluates full MMLU, matching OpenUnlearning's upstream default.
The Bio manifest uses the repository's existing `mmlu_no_bio` task:

```bash
lilpipe configs/lilpipe/experiments/wmdp-bio-rmu.yml --dry-run
```

Bio training requires access to the gated `cais/wmdp-bio-forget-corpus`. The job
reads its cached Parquet snapshot directly and uses the public Bio retain JSONL.
