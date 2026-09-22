# Orthogonal circuit breaker

`di-6.9b.yml` trains a rank-8 adapter on 1,024 WMDP-Bio forget documents and
1,024 WikiText retain documents. The trainer preserves retain activations,
reroutes forget activations away from the base model, and orthogonalizes forget
representations. Run `lilpipe configs/experiments/di-6.9b.yml`; the adapter and
evaluations appear under `artifacts/models/` and `artifacts/evals/`.
