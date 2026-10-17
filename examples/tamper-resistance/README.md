# Tamper Resistance

## Prerequisites

- Request access to [`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
- Update setups in the runner scripts `scripts/slurm/tamia`

## Quickstart

Run these commands from `examples/tamper-resistance`. Training and evaluation use
different uv environments because their dependency groups are not compatible.

```sh
# 0. Setup
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --locked --group train
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --locked --group eval

# 2. Run GradDiff unlearning
source .venv-train/bin/activate
axolotl train configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--2pp--unlearn-fft-gd/selected.yml

# 3. Run relearning
source .venv-train/bin/activate
axolotl train configs/stages-wmdp/2_relearn/z7b-wmdp-bio--2pp--unlearn-fft-gd--relearn-fft/selected.yml

# 4. Run evals
source .venv-eval/bin/activate
CHECKPOINT=artifacts/models/z7b-wmdp-bio--2pp--unlearn-fft-gd--relearn-fft/checkpoint-16
OUTPUT=artifacts/evals/thresholds/2pp/z7b-wmdp-bio--2pp--unlearn-fft-gd--relearn-fft/checkpoint-16
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$CHECKPOINT,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust,mmlu_no_bio \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path $OUTPUT

# 5. Plot the Zephyr-7B relearning trajectory after the jobs finish
# Note: run the above eval for every ckpt in the relearning trajectory before
# running the plotting code
source .venv-eval/bin/activate
python scripts/analysis/plot_trajectory.py \
  --model z7b--2pp \
  --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--2pp.pdf
```

## Reproduce Figure 1. using lilpipe


```sh
# 1. Run all unlearning and tamper resistance methods and evals
# for the Zephyr-7B model, on SLURM.
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--2pp.yml

# 2. Plot the relearning trajectory
source .venv-eval/bin/activate 
python scripts/analysis/plot_trajectory.py \
  --model z7b--2pp \
  --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--2pp.pdf
```

## Hyperparameter optimization

Generate and submit the GD sweep from this directory:
```sh
python scripts/hpo.py generate \
  configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--10pp--unlearn-fft-gd/sweep.yml

lilpipe artifacts/generated/hpo/z7b-wmdp-bio--10pp--unlearn-fft-gd-hpo/experiment.yml
```


## Project structure
```text
configs/
├── experiments/z7b-wmdp-bio-unlearn--2pp.yml  # models and evaluations to run
├── registries/                               # model producers and evaluation jobs
└── stages-wmdp/0_unlearn/z7b-wmdp-bio--2pp--unlearn-fft-gd/
    ├── base.yml                              # training config used by the sweep
    ├── sweep.yml                             # HPO parameter grid
    └── selected.yml                          # optimal training config found
```
