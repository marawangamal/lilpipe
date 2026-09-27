---
name: show-status
description: Use when the user asks for status, progress, or "show status" of lilpipe experiments (e.g. tamper-resistance, weight-steering) — Slurm train/relearn/eval jobs, completed lm-eval metrics, or what is pending for a given method or model (e.g. deff, npo, gd, ws). Builds a metrics table per model with job IDs in place of unfinished cells.
---

# Show Status

Report the end-to-end state of a method's pipeline (unlearn train -> relearn ->
evals) as a single table. Derive everything from configs and artifacts — never
assume a stage exists.

## Workflow

1. **Identify the method's model slugs.** Find the trainer/method name in
   `configs/registries/models.yml` (producer entries) and the experiment files
   in `configs/experiments/*.yml` that list the slugs and their evaluations.
   A slug like `z7b-wmdp-bio-lora-unlearn-deff` implies stages by suffix:
   plain = unlearn, `-relearn` = relearn. Verify each stage actually has a
   registry entry, a config file, and artifacts — report "not set up" when a
   stage the user expects is missing (do not silently drop it).

2. **List queue jobs with full names.** Default `squeue` output truncates the
   name and hides the model. Use:

   ```bash
   squeue --user=$(whoami) -o "%i|%t|%j" --sort=jobid
   ```

   Job names encode the model:
   - `eval-bio-mcqa-ckpts-max<M>-freq<F>-mila-<model-id>[_<suffix>]`
   - `eval-mmlu-no-bio-ckpts-max<M>-freq<F>-mila-<model-id>[_<suffix>]`
   - `train-<model-id>` / `train-<model-id>-relearn`

   Array suffixes `_[a-b]` show which tasks remain queued; tasks no longer in
   the queue have finished (or failed — check artifacts).

3. **Check training artifacts.** `artifacts/mila/models/<model-id>/`:
   - `checkpoint-<step>` dirs up to `max_steps` -> training done
   - `merged/` with full weight shards -> merge done (for `--merge` producers)

4. **Collect finished eval metrics.** Evals live under
   `artifacts/mila/evals/<model-id>/`:
   - checkpoint sweep: `checkpoint-<N>/<task-dir>/<sanitized-model-path>/results_*.json`
   - single model: `<task-dir>/<sanitized-model-path>/results_*.json`
   - task dirs: `wmdp-bio-robust` (key `wmdp_bio_robust`), `mmlu-no-bio`
     (key `mmlu_no_bio`); the metric is `acc,none` in the JSON `results` dict
   - the base model's reference evals may live under a short dir (e.g.
     `artifacts/mila/evals/z7b/`) — check for it

   Checkpoint sweep mapping: array task `i` = checkpoint `i * freq` (freq/max
   come from the eval args in `configs/registries/evals.yml`).

5. **Read metric definitions.** `configs/results/*.yml` defines the table
   columns: label, direction (`minimize`/`maximize`), format (percent),
   precision, task, key, and file globs. Mirror those columns.

## Output format

One markdown table:

- Rows: base model first (for reference), then the method's checkpoints or
  final model, then relearn variants.
- Columns: the metric columns from the results config (e.g. `WMDP-Bio ↓`,
  `MMLU-NoBio ↑`), formatted as percentages at the config's precision.
- Finished cell: the metric value.
- In-flight cell: the specific array task job id (e.g. `10941306_3`) with its
  state, e.g. `10941306_3 (PD)`.
- Missing stage: an explicit `— not set up —` row or note.

After the table, add only high-value notes: parent array job ids and their
coverage, missing stages, and a one-line early signal if finished metrics
already deviate from (or are flat vs) the base model.

Run all commands from the example root (e.g. `examples/tamper-resistance/`),
since artifact paths resolve from the CWD.
