# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Metric Interpretation

**orth_loss**: Average pairwise cosine similarity between forget item representations in each batch. Measures whether forget items collapse to the same direction or remain diverse.
- **~1.0** = bad (all forget items aligned to same direction, vulnerable to single-direction attacks)
- **~0.5** = moderate diversity
- **~0** = good (orthogonal forget directions)

## Results

### LoRA r=16, orth ramp + seq_len scaling

Changes: orth_coeff ramps 0 to full over training (was decaying), orth_loss scaled by mean seq_len to compensate for mean-pooling gradient dilution.

- remove=23, retain=2, r=16, 32 steps, 1024 examples.

| remove | orth | retain | steps | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|-------------|------|-------------|---------|-----------|-------|
| -      | -    | -      | -     | 42.97%      | 45.10% | -         | -       | -         | Baseline |
| 23     | 5    | 2      | 32    | 37.44%      | 43.75% | 21.77     | 0.51    | 730.0     | orth_loss now seq_len-scaled |
| 23     | 20   | 2      | 32    | 38.48%      | 43.65% | 29.64     | 0.49    | 932.5     | |
| 23     | 50   | 2      | 32    | **37.90%**      | **43.55%** | 31.11     | 0.63    | 986.4     | |

#### LoRA has no Tamper Resistance: (retain=2, r=16)

SFT attack (512 examples, lr=2e-5, eval every 5 steps) on the models above.

| Model | Step 0 | Step 5 | Step 10 | Step 25 | Step 50 | Notes |
|-------|--------|--------|---------|---------|---------|-------|
| Baseline | 42.97% | - | - | - | - | Original model |
| orth=5, ret=2 | 37.67% | 42.40% | 39.75% | 41.24% | 43.66% | Full recovery by step 5 |
| orth=20, ret=2 | 38.71% | 41.71% | 41.36% | 42.74% | 44.24% | Full recovery by step 5 |
| orth=50, ret=2 | 38.13% | 41.24% | 41.01% | 42.74% | 44.47% | Full recovery by step 5 |

Plots: `experiment_logs/tamper_orthfix_ret2_mcqa.png`, `experiment_logs/tamper_orthfix_ret2_cloze.png`

### SFT

| remove | orth | retain | steps | WMDP | WMDP Robust | MMLU | retain_loss | cb_loss | orth_loss | Notes |
|--------|------|--------|-------|------|-------------|------|-------------|---------|-----------|-------|
| 23     | 10   | 2      | 32    | -      | -         | -           | -       | -         | SFT planned, keyword mask (type: regex) |
| 30     | 15   | 2      | 32    | 26.55% | -         | -    | 4.30        | 0.18    | 0.99      | |
| 30     | 15   | 2      | 512   | 24.12% | 25.81%    | 23.76% | 1.08        | 0.015   | 0.05      | Severe capability damage |
| 40     | 20   | 2      | 32    | 26.63% | -         | -    | -           | -       | -         | Log truncated |
| 15     | 15   | 2      | 512   | 23.88% | 25.35%    | 29.90% | 1.01        | 0.03    | 0.04      | Capability damage |
| 30     | 100  | 2      | 32    | 39.2%  | -         | -    | 3.52        | 0.35    | 0.33      | Preserved capability, worse unlearning |
| 30     | 100  | 2      | 256   | 29.85% | 28.34%    | 42.18% | 2.95        | 0.10    | 0.09      | |
| 30     | 15   | 2      | 64    | 27.73% | -         | -    | 3.46        | 0.06    | 0.52      | near-baseline MMLU |
| 30     | 15   | 2      | 128   | 24.35% | -         | -    | 2.15        | 0.02    | 0.25      | Better orth, some capability loss |
| 15     | 15   | 15     | 512   | 28.28% | 28.11%    | 43.53% | 0.40        | 0.07    | 0.13      | Good unlearning + MMLU preserved |
| 15     | 15   | 50     | 512   | 35.90% | **33.64%**    | **44.81%** | 0.26        | 0.09    | 0.25      | Better MMLU, weaker unlearning |

### SFT orth CB tamper: rm15 orth15 ret50 (AdamW, 30 epochs, eval every 100)

Model: `deep-ignorance-unfiltered_rm15_orth15_ret50_512`. SFT unlearning with rm=15, orth=15, retain=50, 512 steps.

Tamper: 512 examples, lr=2e-5, 30 epochs (~3300 steps), eval_cloze_prob + eval_mmlu. Job 2476151.

| Metric | Step 0 | Step 100 | Step 200 | Step 300 | Step 400 | Step 500 | Step 600 | Step 700 | Step 800 | Step 900 | Step 1000 | Step 1100 | Step 1200 | Step 1300 | Step 1400 | Step 1500 | Step 1600 | Step 1700 | Step 1800 | Step 1900 | Step 2000 | Step 2100 | Step 2200 | Step 2300 | Step 2400 | Step 2500 | Step 2600 | Step 2700 | Step 2800 | Step 2900 | Step 3000 | Step 3100 | Step 3200 | Step 3300 |
|--------|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| WMDP | 34.10 | 43.78 | 43.20 | 43.78 | 44.59 | 43.78 | 42.63 | 42.86 | 41.36 | 40.90 | | | | | | | | | | | | | | | | | | | | | | | | |
| MMLU | 45.61 | 45.38 | 45.68 | 46.23 | 46.62 | 46.43 | 46.49 | 46.60 | 46.70 | 46.43 | | | | | | | | | | | | | | | | | | | | | | | | |
