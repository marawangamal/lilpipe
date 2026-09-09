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
