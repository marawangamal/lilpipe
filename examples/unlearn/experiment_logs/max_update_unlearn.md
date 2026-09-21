# Max Update Unlearning

Model: EleutherAI/deep-ignorance-unfiltered, SFT

## Results

Jobs 2134097-2134100: update_norm was 0 due to /N normalization bug (dividing by ~3.9B params). Fixed.
Jobs 2270984-2270988: Used squared gradient (positive feedback loop bug). Gradient = -2*coef*(theta-theta_0), grows with displacement.

## Normalized gradient (fixed)

Gradient = -update_coef * (theta-theta_0) / ||theta-theta_0||, constant magnitude.

| Run | retain_coef | update_coef | lr | num_examples | epochs | steps | sft_loss (final) | update_norm (final) | WMDP Bio Robust | MMLU |
|-----|-------------|-------------|------|--------------|--------|-------|------------------|---------------------|-----------------|------|
| Baseline | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 |
| Job 2323707a | 1.0 | 0.01 | 2e-4 | 1024 | 10 | 320 | 0.0177 | 1120446 | 0.2684 | 0.2317 |
| Job 2323707b | 1.0 | 0.1 | 2e-4 | 1024 | 10 | 320 | 3.0779 | 3098849 | 0.2627 | 0.2369 |
| Job 2323707c | 1.0 | 1.0 | 2e-4 | 1024 | 10 | 320 | 6.8090 | 5262352 | 0.2408 | 0.2689 |
| Job 2323707d | 1.0 | 10.0 | 2e-4 | 1024 | 10 | 320 | 7.8545 | 5819048 | 0.2673 | 0.2295 |
| ep1-a | 1.0 | 0.01 | 2e-4 | 1024 | 1 | 32 | 2.1048 | 11848 | 0.2984 | 0.3643 |

## SGD update + element_norm (fixes: grad_acc scaling, per-element normalization, SGD for update term)

| Run | retain_coef | update_coef | lr | num_examples | epochs | steps | sft_loss (final) | update_norm (final) | WMDP Bio Robust | MMLU |
|-----|-------------|-------------|------|--------------|--------|-------|------------------|---------------------|-----------------|------|
| Baseline | - | - | - | - | - | - | - | - | 0.4297 | 0.4510 |
| sgd-a | 1.0 | 0.01 | 2e-4 | 1024 | 10 | 320 | 0.0108 | 19277 | 0.3203 | 0.3781 |
| sgd-b | 1.0 | 0.1 | 2e-4 | 1024 | 10 | 320 | 0.0133 | 71590 | 0.2558 | 0.2593 |
| sgd-c | 1.0 | 1.0 | 2e-4 | 1024 | 10 | 320 | 6.2964 | 6581872 | 0.2673 | 0.2295 |
| sgd-d | 1.0 | 10.0 | 2e-4 | 1024 | 10 | 320 | 549.17 | 748727488 | 0.2569 | 0.2551 |
