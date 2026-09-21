# MAGIC Attribution: WikiText → WMDP-bio-robust

SLURM job 2450464, 2026-02-23.

## Setup

- **Model**: EleutherAI/deep-ignorance-unfiltered (6.86B params)
- **Training data**: 1000 WikiText-103 sequences (indices 186873–187873, ~427–428 chars each)
- **Training**: 250 SGD steps, lr=1e-4, batch_size=4
- **Eval data**: 868 WMDP-bio-robust MCQs (answer-only loss on correct letter token)
- **Hardware**: 4x NVIDIA GH200 120GB, simple_fsdp
- **Torch patch**: pytorch/pytorch#160509 (twice-differentiable DTensor redistribution)

## Timing

| Phase | Time | Peak GPU (per device) |
|-------|------|-----------------------|
| Training (250 steps) | 542.1s | 3.5 GB |
| Eval (868 questions, 109 chunks) | ~60s | 7.0 GB |
| Backward (250 checkpoint replays) | 510.8s | 43.1 GB |
| **Total** | **~18.5 min** | |

## Score Distribution

| Metric | Value |
|--------|-------|
| Shape | [1000] |
| Range | [-3.02e-06, 7.41e-06] |
| Mean | 3.56e-07 |
| Std | 1.06e-06 |
| Negative | 391 |
| Positive | 609 |
| Zero | 0 |

WMDP-bio-robust avg loss: 0.0109

## Top 10 Highest Scores (training increased WMDP-bio eval loss most)

| Rank | Index | Score | Text |
|------|-------|-------|------|
| 1 | 970 | 7.41e-06 | X-Files episode — FBI agent, bank holdup, bomb detonation |
| 2 | 884 | 6.22e-06 | Horror film plot — ghost, murder, body count |
| 3 | 694 | 5.97e-06 | Imperial War Museum founding — 1917 government proposal |
| 4 | 914 | 5.49e-06 | Moreton Old Hall architecture — fireplace, coat of arms |
| 5 | 930 | 5.27e-06 | Portuguese military ambush — officers slain in engagement |
| 6 | 365 | 4.80e-06 | Indus Valley Civilization — Shaktism and Vedic Age |
| 7 | 381 | 4.79e-06 | Viking-era Orkney — Harald captured, ransom in gold |
| 8 | 708 | 4.56e-06 | Second Matabele War — Burnham kills the Mlimo |
| 9 | 679 | 4.50e-06 | El Greco's painting style — use of light |
| 10 | 369 | 4.28e-06 | Heseltine biography — concert reviewing, drinking anthology |

## Top 10 Lowest Scores (training decreased WMDP-bio eval loss most)

| Rank | Index | Score | Text |
|------|-------|-------|------|
| 1 | 998 | -3.02e-06 | 2009 U.S. Open Cup Final — soccer attendance stats |
| 2 | 335 | -2.79e-06 | Gene duplication and mutation — new gene evolution |
| 3 | 997 | -2.46e-06 | Red Auerbach coaching career — basketball history |
| 4 | 717 | -2.24e-06 | Schizophrenia suicide rates and risk factors |
| 5 | 375 | -2.21e-06 | Chhinnamasta shrine in Nepal's Kathmandu Valley |
| 6 | 804 | -2.17e-06 | Boston slave trade history and abolitionism |
| 7 | 485 | -1.99e-06 | Óscar Arias biography — Yale, Sorbonne education |
| 8 | 400 | -1.84e-06 | House martin flea parasites — ectoparasite study |
| 9 | 738 | -1.83e-06 | Hurricane damage — $1.1B Congressional estimate |
| 10 | 818 | -1.81e-06 | ASL as first language in various countries |

## Output Files

| File | Size | Description |
|------|------|-------------|
| `attribution_scores.pt` | 5.5 KB | 1000 float scores |
| `all_scores.json` | 498 KB | All scores with text excerpts |
| `highest_attribution.json` | 27 KB | Top 50 highest scores |
| `lowest_attribution.json` | 27 KB | Top 50 lowest scores |
| `eval_grads.pt` | 3.2 GB | Intermediate resume artifact (param gradients from eval) |
