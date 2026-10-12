# Reproducing GD+SAM tamper resistance in this example

This note records what was required to reproduce the qualitative GD+SAM result
from the [SAM unlearning paper](https://arxiv.org/pdf/2502.05374) in the
Zephyr-7B WMDP-bio experiment. The target behavior was:

1. GD and GD+SAM should both begin with effective unlearning and usable general
   capabilities.
2. Under the same relearning attack, GD+SAM should recover WMDP-bio capability
   more slowly than plain GD.

All WMDP numbers below use `wmdp_bio_robust`; all MMLU numbers use
`mmlu_no_bio`. Lower WMDP is better for unlearning and resistance, while higher
MMLU indicates better retained general capability.

## Final result

The final controlled `ours -> ours` experiment differed only in whether SAM was
used during unlearning. Both methods used the same data processing, sampling,
model placement, seed, and relearning attack.

| Step | Attack samples | WMDP GD | WMDP GD+SAM | MMLU GD | MMLU GD+SAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 28.00 | 31.57 | 48.70 | 52.51 |
| 4 | 16 | 51.96 | 38.36 | 58.12 | 50.26 |
| 8 | 32 | 55.30 | 32.95 | 58.12 | 44.92 |
| 12 | 48 | 55.07 | 38.02 | 57.98 | 50.70 |
| 16 | 64 | 54.95 | 39.86 | 57.99 | 51.67 |

GD+SAM begins close to the paper reproduction's WMDP value of 32 and remains
substantially below GD at every relearning checkpoint. The non-monotonic values,
including the MMLU dip at step 8, are the measured trajectory rather than a
smoothed summary.

The canonical configs that produce this comparison are:

- `examples/tamper-resistance/configs/stages/0_unlearn/z7b-wmdp-bio--unlearn-fft-gd/selected.yml`
- `examples/tamper-resistance/configs/stages/0_unlearn/z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2/selected.yml`
- `examples/tamper-resistance/configs/stages/2_relearn/z7b-wmdp-bio--unlearn-fft-gd--relearn-fft/selected.yml`
- `examples/tamper-resistance/configs/stages/2_relearn/z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2--relearn-fft/selected.yml`

They are selected together by
`examples/tamper-resistance/configs/experiments/unlearn-relearn.yml`.

### Fresh canonical reproduction

After deleting the prior model artifacts, the canonical selected configs were
rerun from scratch on Tamia. The training/relearning jobs were `612585` through
`612588`; evaluation arrays were `612589` through `612592`. Every job completed
with exit code `0:0`.

| Step | Attack samples | WMDP GD | WMDP GD+SAM | MMLU GD | MMLU GD+SAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 28.00 | 31.57 | 48.70 | 52.51 |
| 4 | 16 | 52.07 | 38.36 | 58.17 | 50.26 |
| 8 | 32 | 54.84 | 32.95 | 58.13 | 44.92 |
| 12 | 48 | 55.18 | 38.02 | 58.02 | 50.70 |
| 16 | 64 | 55.18 | 39.63 | 57.95 | 51.69 |

This independently reproduces the intended qualitative behavior: both methods
start with useful MMLU capability and low WMDP, while GD+SAM remains 15.55 WMDP
points below GD after 64 attack samples. The small differences from the earlier
table do not change that conclusion.

### Ablation: mask padding from the loss

The canonical experiment intentionally reproduces the authors' processing,
which copies padded EOS token IDs into `labels`. A controlled ablation changed
only the unlearning processors to the standard implementations, where padded
labels are `-100` and therefore ignored by cross-entropy. Sampling, seeds,
optimization, model placement, and the relearning attack were unchanged.

The ablation training/relearning jobs were `612605` through `612608`; evaluation
arrays were `612609` through `612612`. Every job completed with exit code `0:0`.

| Step | Attack samples | WMDP GD | WMDP GD+SAM | MMLU GD | MMLU GD+SAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 26.96 | 54.15 | 46.09 | 58.65 |
| 4 | 16 | 27.30 | 55.18 | 23.02 | 58.16 |
| 8 | 32 | 26.73 | 53.46 | 22.92 | 58.22 |
| 12 | 48 | 26.73 | 53.69 | 22.92 | 58.20 |
| 16 | 64 | 26.73 | 53.34 | 22.92 | 58.18 |

With padding masked, GD+SAM retains MMLU but fails to unlearn WMDP at step 0.
GD initially unlearns, but the nominal relearning attack collapses MMLU to near
chance without recovering WMDP. Thus the standard-processing variant does not produce
a valid GD-versus-GD+SAM resistance comparison under these hyperparameters.
Including padded EOS tokens is empirically essential to reproducing the selected
result as configured; this ablation does not establish that including padding is
generally preferable to conventional masking.

## What fixed initial unlearning

The corpus was not the discrepancy. The local OPTML WMDP forget corpus and the
Hugging Face `cais/wmdp-bio-forget-corpus` copy had the same 24,453 raw rows,
the same 24,432 rows after filtering texts of length at most 50, and identical
content and order. The WikiText splits were also identical.

The important difference was sampling. OPTML uses PyTorch `RandomSampler` for
the paired forget examples and chooses retain examples dynamically, with
replacement, in `__getitem__`. Our earlier implementation shuffled with a
separate `random.Random(seed)` instance and precomputed the retain draws. These
procedures do not generate the same batches even when given the same nominal
seed.

There was also a seed mismatch hidden in the authors' setup. Although the
experiment configuration stated `overall.seed: 3`, OPTML did not pass that
value to `TrainingArguments`, so Hugging Face Trainer reset the relevant RNGs
using its default seed of 42. The canonical configs therefore use:

```yaml
custom_trainer_args:
  sampling: optml
seed: 42
data_seed: 42
```

Matching this sampling behavior changed GD+SAM from the original ineffective
initial WMDP of 54.95 to an effectively unlearned checkpoint. Sampling was the
main fix for the initial checkpoint.

## What fixed relearning resistance

Sampling parity alone was not sufficient. The sampler-matched GD+SAM run began
at WMDP 26.73, but recovered to 47.81 after only 16 attack samples and remained
around 50 thereafter. It had effective unlearning but not the desired
resistance.

Resistance appeared after matching the rest of the OPTML execution behavior as
a bundle:

- Authors-style document processing, including padded EOS tokens in the loss.
- OPTML's original-dtype BF16 SAM gradient-norm calculation.
- Eager attention.
- Deterministic execution for the SAM run.
- The authors' four-GPU model-parallel layer placement.
- SAM radius `rho: 0.01`.
- The complete 16-step relearning attack and its linear learning-rate schedule.

The model-parallel and SAM-norm fixes are important for correctness: a SAM norm
computed independently from only a shard does not represent the perturbation
used by the full model. For strict OPTML parity, the implementation also needs
to retain the authors' original BF16 norm behavior rather than silently
switching to the newer full-model FP32 norm implementation.

The full attack schedule matters when interpreting diagnostics. A four-step
screen with `max_steps: 4` used learning rates of `1e-5`, `7.5e-6`, `5e-6`, and
`2.5e-6`. The first four steps of the intended 16-step attack instead used
`1e-5`, `9.375e-6`, `8.75e-6`, and `8.125e-6`. Consequently, the four-step
screen was not a like-for-like estimate of checkpoint 4 in the full attack.

We did not run a complete factorial ablation of every item in this second
bundle. The experiments establish that sampling fixed the initial unlearning
failure, that sampling alone did not recover resistance, and that the complete
parity bundle did. They do not justify attributing the resistance improvement
to any single remaining setting.

## Negative result: increasing rho

Increasing the SAM radius from 0.01 to 0.05 did not improve resistance. It
destroyed initial unlearning efficacy:

| Step | Attack samples | WMDP GD+SAM, rho=0.05 |
| ---: | ---: | ---: |
| 0 | 0 | 53.11 |
| 4 | 16 | 55.76 |
| 8 | 32 | 55.76 |
| 12 | 48 | 55.65 |
| 16 | 64 | 55.65 |

The retained canonical radius is therefore 0.01.

## Running the canonical experiment

All paths in lilpipe resolve from the current working directory. Run the
pipeline from the example root:

```bash
cd examples/tamper-resistance
lilpipe configs/experiments/unlearn-relearn.yml --dry-run
lilpipe configs/experiments/unlearn-relearn.yml
```

The pipeline trains the two unlearned models, applies the identical 16-step
relearning attack, saves checkpoints every four steps, and evaluates the five
points at 0, 16, 32, 48, and 64 attack samples.

## Summary

The shortest accurate explanation is:

- Exact OPTML sampling and seed behavior fixed initial unlearning.
- Matching the remaining SAM execution details and evaluating the correct full
  attack schedule recovered relearning resistance.
- A larger SAM radius was not a substitute for parity and failed badly.
