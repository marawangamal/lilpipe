# Tamper Resistance

## Prerequisites

- Request access to [`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
- Update setups in the runner scripts `scripts/slurm/tamia`

## Quickstart

```sh
# Put artifacts and Slurm logs on scratch
mkdir -p "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts/tamia/logs"
ln -sfnT "$SCRATCH/lilpipe/examples/tamper-resistance/artifacts" artifacts

# Run GD, GD+SAM, RMU, RMU+SAM, and other Zephyr-7B lineages
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--2pp.yml

# Plot the Zephyr-7B relearning trajectory after the jobs finish
UV_PROJECT_ENVIRONMENT="$SCRATCH/lilpipe/examples/tamper-resistance/.venv-eval" \
uv run --group eval python scripts/analysis/plot_trajectory.py \
  --model z7b--2pp \
  --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--2pp.pdf
```

## Project structure
```text
configs/
├── experiments/z7b-wmdp-bio-unlearn--2pp.yml  # models and evaluations to run
├── registries/                               # model producers and evaluation jobs
└── stages/0_unlearn/z7b-wmdp-bio--2pp--unlearn-fft-gd/
    ├── base.yml                              # training config used by the sweep
    ├── sweep.yml                             # HPO parameter grid
    └── selected.yml                          # optimal training config found
```

## Hyperparameter optimization

Generate and submit the GD sweep from this directory:
```sh
uv run python scripts/hpo.py generate \
  configs/stages/0_unlearn/z7b-wmdp-bio--10pp--unlearn-fft-gd/sweep.yml

lilpipe artifacts/generated/hpo/z7b-wmdp-bio--10pp--unlearn-fft-gd-hpo/experiment.yml
```
