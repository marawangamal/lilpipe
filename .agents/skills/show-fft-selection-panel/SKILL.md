---
name: show-fft-selection-panel
description: Show the selected Zephyr-7B full-finetuning GD and GD+SAM unlearning/relearning results, HPO artifact paths, and canonical promotion state. Use when the user asks for the FFT selection panel or a concise paper-style summary of selected FFT experiments.
---

# Show the FFT selection panel

Render the current selected full-finetuning results from artifacts rather than
repeating remembered metric values. This workflow is read-only: do not copy,
delete, promote, submit, or modify jobs.

## Selected runs

Unless the user identifies a newer selection, use these Tamia model slugs:

- GD unlearn: `z7b-wmdp-bio-fft-s125-unlearn-gd-hpo-lr2p5e-6`
- GD relearn: `z7b-wmdp-bio-fft-s125-unlearn-gd-relearn-hpo-lr2p5e-6`
- GD+SAM unlearn: `z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3-hpo-lr2p5e-6`
- GD+SAM relearn: `z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3-relearn-hpo-lr2p5e-6`

The selected GD+SAM rho is `1e-3`. Relearning is ordinary fine-tuning, so it
does not have a second rho.

## Collect current evidence

Run queries from `examples/tamper-resistance` on the `tamia` SSH host.
Artifacts live under `artifacts/tamia`.

For each selected slug:

1. Read all recursive `results*.json` files under
   `artifacts/tamia/evals/<slug>`.
2. Extract `results.wmdp_bio_robust["acc,none"]` and
   `results.mmlu_no_bio["acc,none"]`, expressed as percentages.
3. Verify the corresponding `artifacts/tamia/models/<slug>` and
   `artifacts/tamia/evals/<slug>` paths exist. Report missing evidence
   explicitly instead of substituting remembered values.

Check these canonical paths independently and list only paths that actually
exist:

- `artifacts/tamia/models/z7b-wmdp-bio-fft-s125-unlearn-gd`
- `artifacts/tamia/models/z7b-wmdp-bio-fft-s125-unlearn-gd-relearn`
- `artifacts/tamia/models/z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3`
- `artifacts/tamia/models/z7b-wmdp-bio-fft-s125-unlearn-gd-sam-rho1e-3-relearn`
- the equivalent paths under `artifacts/tamia/evals/`

## Output

Return, in order:

1. A compact paper-style table with columns: Method, rho, Stage, LR,
   WMDP Bio down, and MMLU-no-bio up.
2. A list of the four selected HPO model/evaluation artifact pairs.
3. A list of canonical promoted paths that exist.
4. One short note naming selected artifacts that have not been promoted.

Use two decimal places for percentages. Do not include job history or other
rho values unless requested.
