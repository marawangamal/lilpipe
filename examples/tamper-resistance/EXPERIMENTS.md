# Experiments

## Investigate whether Persona/Character training and Constitutional AI is effective for Tamper Resistance

We want to evaluate SmoLM3-{TR-METHOD} models on TamperBench. We wil try a number of methods, for now let's understand the base model's tamper resistance performance.

## LAT weight steering

`unfiltered-weight-steering.yml` tests adapter arithmetic as an unlearning
intervention on `EleutherAI/deep-ignorance-unfiltered`. Matched SFT arms learn
the LAT `chosen` and `rejected` completions, and the final PEFT adapter applies
`alpha * (chosen - rejected)` using PEFT's concatenation composition. The sign
therefore moves toward harmless responses and away from harmful responses.

Alpha 1, alpha 2, and alpha 10 are evaluated on zero-shot `wmdp_bio_robust`:
alpha 1 uses the contrastive direction at its learned magnitude, while the
larger values test extrapolation. The model registry expresses both arms as dependencies,
so selecting the final models schedules the full training/build/evaluation DAG.
Run `lilpipe configs/experiments/unfiltered-weight-steering.yml`; adapters and
evaluations appear under `artifacts/models/` and `artifacts/evals/`.
