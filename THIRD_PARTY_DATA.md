# Third-party data provenance

The Python package itself contains no runtime datasets. The tracked files under
`examples/weight-steering/data/` and `examples/weight-steering/lm_eval_tasks/`
are example research inputs and are not relicensed by the repository's MIT
license.

- `data/deception/mbpp_honeypot_train.jsonl` and
  `lm_eval_tasks/deception/data/mbpp_evalplus.jsonl` are derived from MBPP and
  EvalPlus/MBPP+ research data. Users must comply with the source datasets'
  licenses and terms.
- `data/qwen36-27b-on-policy-honesty/` contains generated research data; its
  local README documents its generation provenance.
- `data/sycophancy-control/` contains generated control examples used by the
  bundled experiment.

These files are included for reproducibility. Verify upstream terms before
redistributing or using them outside the examples.

The TOFU pilot under `examples/weight-steering` downloads `locuslab/TOFU`
directly from Hugging Face at run time; no TOFU rows are committed here. TOFU
is released under the MIT license. The evaluator ports metric definitions from
the OpenUnlearning repository at revision
`4ad738aaf60f6a4385f6e2506d01da99e76c31f3` and records that revision in every
result artifact.

The robust WMDP-Bio lm-eval task definitions under
`examples/tamper-resistance/lm_eval_tasks/wmdp_bio_categorized_mcqa/` are
vendored from EleutherAI's `deep-ignorance` repository at commit
`1d542e35aacdc0ab3592cfdb978a9d90d2624a66`. They reference the hosted
`EleutherAI/wmdp_bio_robust_mcqa` dataset; no evaluation examples are copied
into this repository. The gated `cais/wmdp-bio-forget-corpus` training data is
downloaded only at run time and stored beneath the ignored experiment
`artifacts/data/` directory.
