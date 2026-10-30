# Tamper Resistance

## Prerequisites

- Request access to [`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
- Update setups in the runner scripts `scripts/slurm/tamia`

## Quickstart

```sh
# 0. Setup
cd examples/tamper-resistance
# NOTE: Training and eval use diff uv envs due to incompatible dependencies
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --locked --group train
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --locked --group eval

# 2. Run a selected 5pp unlearning method
source .venv-train/bin/activate
axolotl train configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu/selected.yml

# 3. Run relearning
source .venv-train/bin/activate
axolotl train configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft/selected.yml

# 4. Run eval on final ckpt
source .venv-eval/bin/activate
CHECKPOINT=artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft/checkpoint-16
OUTPUT=artifacts/tamia/evals/z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft/checkpoint-16
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
python scripts/analysis/plot_trajectory_wmdp.py \
  --plot main \
  --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf
```

## Reproduce Figure 1. using lilpipe


```sh
# 1. Run all unlearning and tamper resistance methods and evals
# for the Zephyr-7B model, on SLURM.
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--5pp.yml

# 2. Plot the relearning trajectory
source .venv-eval/bin/activate 
python scripts/analysis/plot_trajectory_wmdp.py \
  --plot main \
  --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf
```

## Hyperparameter optimization for Qwen3-0.6B experiments

Run from `examples/tamper-resistance`. The defense grid is `[0.01, 0.1, 1.0]`,
corresponding to candidates `beta--001`, `beta--002`, and `beta--003`.
The attack grid crosses those three model paths with LRs `[5e-5, 1e-4]`.
Keep the grid order fixed once an experiment has started.

```sh
# 1. Generate and run the multi-beta defenses and their evaluations.
python scripts/hpo.py generate configs/stages-beavertails/0_alignment/qwen3-0.6b-beavertail--align-fft-seam-legacy/sweep.yml
lilpipe artifacts/generated/hpo/qwen3-0.6b-beavertail--align-fft-seam-legacy--hpo/experiment.yml

# 2. Wait for all defenses to finish; keep their models in the generated grid.
# No model promotion or retraining is needed before relearning.

# 3. Generate attacks, then run one LR pair at a time with its evaluations.
python scripts/hpo.py generate configs/stages-beavertails/1_relearn/qwen3-0.6b-beavertail--align-fft-seam-legacy/sweep.yml
attack_id=qwen3-0.6b-beavertail--align-fft-seam-legacy--relearn--hpo
lilpipe "artifacts/generated/hpo/$attack_id/experiment.yml" \
  --models "$attack_id--beta0.01--001" "$attack_id--beta0.01--002"
# After both evaluations and cleanup succeed, repeat with beta0.1, then beta1.0.

# 4. Plot all three betas and both LRs.
.venv-eval/bin/python scripts/analysis/plot_trajectory_beavertails.py
```

Attack records `beta0.01`, `beta0.1`, and `beta1.0` each generate candidates
`001` (LR `5e-5`) and `002` (LR `1e-4`). Evaluations cover steps 0, 100, and 200.
Each attack trajectory is deleted only after its evaluations succeed; defense
models and evaluation summaries are retained.

The existing runs were reindexed without retraining. Their original configs and
results are archived under `artifacts/generated/hpo/archive/` and
`artifacts/mila/evals/archive/`; reindexed summaries record their original
`source_summary`. Regenerating configs does not skip completed jobs: the submission
commands above are for fresh runs, and should not be rerun over completed artifacts.


## Hyperparameter optimization

Generate and submit an RMU sweep from this directory:
```sh
python scripts/hpo.py generate \
  configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu/sweep.yml

lilpipe artifacts/generated/hpo/z7b-wmdp-bio--5pp--unlearn-fft-rmu-hpo/experiment.yml
```


## Project structure
```text
configs/
├── experiments/z7b-wmdp-bio-unlearn--5pp.yml  # models and evaluations to run
├── registries/                               # model producers and evaluation jobs
└── stages-wmdp/0_unlearn/
    └── z7b-wmdp-bio--5pp--unlearn-fft-rmu/
        ├── base.yml                          # training config used by the sweep
        ├── sweep.yml                         # HPO parameter grid
        └── selected.yml                      # optimal training config found
```
