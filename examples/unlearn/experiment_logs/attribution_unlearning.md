# Attribution-Based Unlearning Experiments

Base model: `EleutherAI/deep-ignorance-unfiltered` (6.86B, Pythia architecture)
Attribution scores: MAGIC per-token backprop through SGD training steps (LR=1e-4, max_seq_len=1024)
Baseline: WMDP Bio Robust=0.4297, MMLU=0.4510

## Available MAGIC Attribution Scores

All scores are per-token tensors of shape [N, 1024] stored as `per_token_scores.pt`.

Note: the full bio-forget/bio-remove corpus has 24,453 docs. The 10k runs below used only 10,000 docs (~41% of the corpus). No attribution run has been done on the full dataset.

| Training Data | N | Tokens/seq | Steps | Eval Task | Output Dir | Status |
|---|---|---|---|---|---|---|
| WikiText-103 | 1000 | 1024 | 250 | WMDP Bio Robust | `magic_wmdp_msl1024_output` | Complete |
| WikiText-103 | 1000 | 1024 | 250 | MMLU | `magic_mmlu_msl1024_output` | Complete |
| bio-retain | 1000 | 1024 | 250 | WMDP Bio Robust | `magic_wmdp_retain_msl1024_output` | Complete |
| UltraChat | 1000 | 1024 | 250 | WMDP Bio Robust | `magic_ultrachat_msl1024_output` | Complete |
| bio-forget | 1000 | 1024 | 250 | WMDP Bio Robust | `magic_wmdp_forget_wmdp_msl1024_output` | Complete |
| wmdp-lie-o | 1000 | 1024 | 250 | WMDP Bio Robust | `magic_wmdp_lie_o_wmdp_msl1024_output` | Complete |
| bio-forget | 10000 | 1024 | 1250 | WMDP Bio Robust | `magic_wmdp_forget_10k_wmdp_msl1024_output` | Complete |
| bio-forget | 10000 | 1024 | 1250 | MMLU | `magic_wmdp_forget_10k_mmlu_msl1024_output` | Complete |
| wmdp-lie-o | 10000 | 1024 | 1250 | WMDP Bio Robust | `magic_wmdp_lie_o_10k_wmdp_msl1024_output` | Complete |

Checkpoints stored at `/projects/a6a/public/lucia/magic_{tag}_msl1024_ckpts/`.
Output dirs under `bergson3/runs/`.

Attribution scores indicate the effect on loss, so a negative score indicates that training on the token will reduce loss. This is what we want!

All attribution scores are taken on the same dataset as the unlearning dataset, using a model trained on the unfiltered/weighted unlearning dataset, unless otherwise stated.

## Failed Experiment 1: Attribution-weighted unlearning (all tokens, value-weighted)

Uses WMDP attribution scores as per-token loss weights (normalized by mean absolute value so typical weight ~ ±1). Negative-score tokens get gradient ascent (unlearn), positive-score tokens get gradient descent (reinforce).

- SFT
- bio-retain corpus
- 1k training examples
- Adam, 1 epoch, bs=4, 250 steps.
- All results invalid due to bad HPs

Script: `bergson3/runs/attribution_unlearn.py`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 5e-6 | 0.4332          | 0.4490 |
| 1e-5 | 0.4378          | 0.4500 |
| 2e-5 | 0.4332          | 0.4481 |
| 5e-5 | 0.3975          | 0.4106 |
| 8e-5 | 0.2523          | 0.2530 |
| 1e-4 | 0.2408          | 0.2689 |
| 2e-4 | 0.2673          | 0.2295 |

## Failed Experiment 2: Attribution sign-only unlearning (all tokens, sign-weighted)

Multiply per-token losses by the sign of WMDP attribution scores, so negatively-attributed/proponent tokens are unlearned and vice versa.

- All results invalid due to bad HPs

Script: `bergson3/runs/attribution_unlearn.py --sign_only`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-7 | 0.4320          | 0.4506 |
| 1e-6 | 0.4343          | 0.4504 |
| 2e-6 | 0.4274          | 0.4490 |
| 5e-6 | 0.3779          | 0.4154 |
| 1e-5 | 0.2673          | 0.2295 |
| 5e-5 | 0.2350          | 0.2465 |
| 8e-5 | 0.2350          | 0.2465 |
| 1e-4 | 0.2350          | 0.2465 |
| 2e-4 | 0.2350          | 0.2465 |

## Failed Experiment 3: Positive-only token training (WikiText, WMDP scores)

Positively-attributed/detractor tokens only (WMDP score > 0). Negative/zero tokens masked with -100.

- All results invalid due to bad HPs
- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=32, 31 steps, max_seq_len=256.

Script: `bergson3/examples/validate_attribution.py --experiment positive_only`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-6 | 0.4297          | 0.4494 |
| 1e-5 | 0.4274          | 0.4484 |
| 5e-5 | 0.3894          | 0.3823 |
| 2e-4 | 0.2385          | 0.2502 |

## Failed Experiment 4: Negative-only token training (WikiText, WMDP scores)

Negative-attributed/proponent tokens only (WMDP score < 0). Positive/zero tokens masked with -100.

- All results invalid due to bad HPs
- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024.

Script: `bergson3/examples/validate_attribution.py --experiment negative_only`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-5 | 0.4343          | 0.4479 |
| 5e-5 | 0.4666          | 0.4575 |
| 1e-4 | 0.4240          | 0.4476 |
| 2e-4 | 0.3548          | 0.3714 |

## Failed Experiment 5: Selective token training (WMDP+ AND MMLU≤0)

Tokens that detract from WMDP (positive attribution score) and are proponents of MMLU (negative attribution score).

These tokens make up 14.8% of the 37,452 in the dataset.

- All results invalid due to bad HPs
- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024.

Script: `bergson3/examples/validate_attribution.py --experiment selective`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-6 | 0.4320          | 0.4484 |
| 1e-5 | 0.4240          | 0.4511 |
| 2e-5 | 0.4147          | 0.4422 |
| 3e-5 | 0.3848          | 0.4103 |
| 4e-5 | 0.3191          | 0.3482 |
| 5e-5 | 0.2811          | 0.3027 |
| 2e-4 | 0.2350          | 0.2467 |

Note: experiments 5 and 6 select the same tokens (no MMLU scores are exactly 0).

## Failed Experiment 6: Compatible token training (WMDP+ AND MMLU-)

Tokens with positive WMDP score (training hurts WMDP) AND negative MMLU score (training helps MMLU).

- All results invalid due to bad HPs
- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024

Script: `bergson3/examples/validate_attribution.py --experiment compatible`

| LR   | WMDP Bio Robust | MMLU |
|------|-----------------|------|
| 1e-5 | 0.4240          | 0.4511 |
| 2e-5 | 0.4147          | 0.4422 |
| 3e-5 | 0.3848          | 0.4103 |
| 4e-5 | 0.3191          | 0.3482 |
| 5e-5 | 0.2811          | 0.3027 |
| 1e-4 | 0.2362          | 0.2499 |
| 2e-4 | 0.2350          | 0.2467 |

## Experiment 7: All tokens baseline (WikiText, no attribution masking)

All tokens (no masking). Baseline to compare against attribution-guided experiments 3-6.

- All results invalid due to bad HPs

This experiment tells us that our underlying training setup is flawed, rendering all previous experiments useless.

- SFT
- WikiText-103
- 1k examples
- AdamW, cosine schedule, warmup=0.1, bs=8, 125 steps, max_seq_len=1024

Script: `bergson3/examples/validate_attribution.py --experiment all_tokens`

| LR   | WMDP Bio Robust | MMLU   |
|------|-----------------|--------|
| 1e-5 | 0.4274          | 0.4469 |
| 5e-5 | 0.4297          | 0.4311 |
| 1e-4 | 0.4194          | 0.4227 |
| 2e-4 | 0.4009          | 0.3824 |
