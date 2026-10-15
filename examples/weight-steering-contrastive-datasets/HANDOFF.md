# Handoff: Fix the PerTA-grad reproduction

## Objective

Determine why our PerTA-grad task-negation implementation performs much worse
than the paper, then make our reproduction work.

- Paper: [Per-parameter Task Arithmetic](https://arxiv.org/pdf/2601.22030v1)
- Benchmark: TOFU `forget10`
- Model: Llama-3.2-1B-Instruct
- Implementation commit: `708932b`

## Current results

Forget Quality is displayed as `log10(raw FQ)`; values closer to zero are
better.

| Source | Method | log10(FQ) | Utility |
| --- | --- | ---: | ---: |
| Ours | Vanilla task negation | -1.999 | 0.432 |
| Ours | PerTA-grad task negation | -12.266 | 0.450 |
| Paper | Vanilla task vector | -12.506 | 0.451 |
| Paper | PerTA-grad | -1.107 | 0.542 |
| Oracle | Retain-only model | 0.000 | 0.591 |

Our outcome is almost the reverse of the paper: vanilla task negation has
surprisingly good FQ, while applying PerTA-grad suppresses too much useful
forgetting and raises utility only slightly.

| Method | Raw FQ | MU | Forget QA probability | ROUGE | Truth ratio | Extraction strength |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours, vanilla task negation | 0.01003 | 0.4319 | 0.2442 | 0.3759 | 0.6585 | 0.06597 |
| Ours, PerTA-grad task negation | 5.419e-13 | 0.4501 | 0.5191 | 0.4399 | 0.5528 | 0.1493 |

The complete generated table is at `artifacts/results/table.md`.

## Implemented formulas

Vanilla task negation:

```text
theta_final = theta_full - (theta_forget_only - theta_0)
```

PerTA-grad:

```text
W = (abs(g_forget) + epsilon)
    / (abs(g_forget) + abs(g_retain) + 2 * epsilon)

theta_final = theta_full - W * (theta_forget_only - theta_0)
```

Relevant files:

- `scripts/perta_mask.py`
- `scripts/steer.py`
- `scripts/train.py`
- `scripts/hpo.py`
- `configs/sweeps.yml`
- `../../tests/test_weight_steering_contrastive_datasets.py`

## Verified training setup

The forget-only model uses:

- initialization: `meta-llama/Llama-3.2-1B-Instruct`;
- TOFU `forget10`, containing 400 examples;
- 250 optimizer steps;
- batch size 2 and gradient accumulation 16;
- effective batch size 32, giving exactly 20 dataset passes;
- learning rate `1e-5`;
- AdamW default weight decay `0.01`;
- answer-only loss, with prompt labels masked;
- final recorded training loss approximately `0.00413`.

The PerTA mask uses:

- gradients evaluated on the pretrained model;
- the complete `forget10` and `retain90` datasets;
- batch size 2 and maximum sequence length 384;
- `epsilon=1e-8`;
- 146 tensors covering 1.236B parameters;
- bfloat16 serialization;
- range `[4.59e-6, 1.0]`, sample mean `0.507`, and sample quantiles
  `q10=0.328`, `q50=0.5`, `q90=0.703`.

## Artifacts

- Forget-only model:
  `artifacts/generated/hpo/models/task-negation-forget-only`
- Full mask:
  `artifacts/generated/hpo/masks/perta-grad-full`
- Vanilla model:
  `artifacts/generated/hpo/models/task-negation-forget10-a1p0`
- Masked model:
  `artifacts/generated/hpo/models/perta-grad-task-negation-forget10-a1p0`
- Full evaluations:
  `artifacts/evals/task-negation-forget10` and
  `artifacts/evals/perta-grad-task-negation-forget10`

All production jobs completed successfully. The focused validation currently
passes 27 tests.

## Highest-priority investigation

1. **Loss tokenization and masking.** Our gradients and forget-only model use
   answer-only loss. Determine whether the paper computes NLL over the complete
   formatted QA sequence, including prompt tokens. Recompute the mask using the
   exact paper or OpenUnlearning collator.

2. **Dataset-gradient aggregation.** We average Hugging Face token-mean batch
   losses, weighted by example count. This is not necessarily equivalent to
   mean sequence NLL, global mean token NLL, or mean per-example gradients.
   Reproduce the paper's precise reduction.

3. **Epsilon.** The paper does not clearly provide its numerical value.
   `1e-8` may force low-gradient parameters toward `0.5`. Test values such as
   `1e-12`, `1e-16`, and effectively zero with safe division handling.

4. **Mask precision.** Test float32 mask serialization to exclude bfloat16
   quantization as a cause.

5. **Optimizer and scheduler.** We use a constant learning rate. The paper
   states that training includes a one-epoch warm-up and uses 20 epochs for the
   forget-only model. Reproduce its exact scheduler.

6. **Reference implementation.** Search for released PerTA code or linked
   configurations. The equations may not specify all preprocessing and loss
   reductions.

7. **Mask diagnostics.** Compare layer-wise mask statistics, task-vector norm
   retained by the mask, correlation between mask and task-vector magnitude,
   and masks estimated at `theta_0` versus `theta_full`.

8. **Controlled ablations.** Evaluate constant masks such as `0.25`, `0.5`,
   and `0.75`, plus a shuffled learned mask. This will reveal whether the
   learned mask does more than globally shrink the task vector.

## Contrastive steering result

PerTA-grad was also applied to the custom contrastive vector

```text
theta_full + alpha * W * (theta_positive - theta_forget)
```

and did not help:

| Method | log10(FQ) | MU |
| --- | ---: | ---: |
| Original retain-arm steering | -17.781 | 0.469 |
| Masked retain-arm steering | -20.468 | 0.469 |
| Masked IDK steering | -18.070 | 0.314 |

Focus first on reproducing paper-style PerTA task negation before revisiting
this hybrid.

## Worktree caution

The repository contains unrelated dirty sandbagging changes. Do not stage,
overwrite, or clean them. The contrastive experiment is committed in
`708932b`; its scratch `artifacts` symlink is intentionally untracked.
