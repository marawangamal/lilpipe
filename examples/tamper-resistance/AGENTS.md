# Tamper Resistance: Agent Guidelines

Run every command from `examples/tamper-resistance`; all paths below are
relative to it. Training uses `.venv-train`, evaluation uses `.venv-eval`
(see `README.md`).

## Utility Budgets

Lineage names carry a utility budget tag, e.g. `z7b-wmdp-bio--5pp--unlearn-fft-rmu`.
`<N>pp` is the maximum allowed MMLU-no-bio drop, in percentage points, from the
base model. For Zephyr-7B the base MMLU-no-bio score is `58.59`:

| Tag | Max MMLU drop | Min MMLU-no-bio |
| --- | ---: | ---: |
| `--5pp--` | 5 | 53.59 |
| `--15pp--` | 15 | 43.59 |

Evaluations for a budget are written under `artifacts/tamia/evals/thresholds/<N>pp/`.

## Hyperparameter Tuning Procedure

1. **Define the sweep.** Each lineage lives at
   `configs/stages-wmdp/0_unlearn/<lineage>/` with a runnable `base.yml` and a
   `sweep.yml`. Each named stage in `sweep.yml` is an independent grid; only
   parameters inside one stage form a Cartesian product.
2. **Generate and submit.**

   ```bash
   python scripts/hpo.py generate configs/stages-wmdp/0_unlearn/<lineage>/sweep.yml
   lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml --dry-run
   lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml
   ```

   Generated configs, the model registry, and the manifest go to
   `artifacts/generated/hpo/<sweep-id>/`. Grid point `--grid--NNN` maps to
   values through `artifacts/generated/hpo/<sweep-id>/configs/`. HPO runs keep
   only the final checkpoint.
3. **Monitor.** Jobs run on Tamia (`ssh tamia`; the checkout is
   `~/projects/lilpipe`). Check them with `squeue -u $USER` and `sacct -u $USER -X`.
   Each grid point gets one training job and one job per evaluation
   (`wmdp-bio-final-<N>pp`, `mmlu-no-bio-final-<N>pp`).
4. **Collect.** For every grid point, read `acc,none` from the lm-eval
   `results_*.json` under
   `artifacts/tamia/evals/thresholds/<N>pp/<sweep-id>--grid--NNN/checkpoint-*/{wmdp-bio-robust,mmlu-no-bio}/`.
5. **Select.** A candidate is *eligible* if its MMLU-no-bio drop stays within
   the lineage's budget (e.g. under 5 points for `--5pp--`). Among eligible
   candidates, select the one with the **lowest WMDP-Bio robust accuracy**,
   i.e. the strongest unlearning. Candidates near chance on WMDP (~25%) usually
   fail the budget because MMLU collapses too; never select them.
6. **Refine if needed.** If the best eligible point sits on the edge of the grid,
   or the jump between neighbouring points is large, e.g. from eligible to
   collapsed, add a finer sweep stage around it and repeat from step 2.
7. **Promote.** Copy the chosen concrete values into the lineage's
   `selected.yml`, promote its model in `configs/registries/models.yml`, and
   record the sweep (grid, run count, promoted setting) in the
   *Hyperparameter sweeps* table of `docs/experiments.md`. When the choice is
   close, also add an eligibility table like the RMU + AO one.
