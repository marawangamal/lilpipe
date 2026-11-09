Earlier noise, norm, and ablation reports: [results history](results-history.md).

| Stream | Authors | Ours before | Ours, refusal space | Ours, harmful + refusal spaces |
|---|---|---|---|---|
| Harmful (HH) | `Answer:\n{harmful answer}` | `Answer:\n{harmful answer}` | `Answer:\n{harmful answer}` | `Answer: {harmful answer}` |
| Refusal (HB) | `Answer: {refusal}` | `Answer:\n{refusal}` | `Answer: {refusal}` | `Answer: {refusal}` |
| Benign (BB) | `Answer: {helpful answer}` | `Answer: {helpful answer}` | `Answer: {helpful answer}` | `Answer: {helpful answer}` |

The refusal-space variant changes only the refusal separator. The new variant
also replaces the harmful separator with a space. Benign formatting is unchanged.
`\n` means a newline.

| Defense variant | Before attack | After LR `1e-4` | After LR `2e-4` |
|---|---:|---:|---:|
| SEAM — our original implementation | 48.42% | 51.40% | 50.13% |
| SEAM — ours, refusal formatting corrected | 51.87% | 29.48% | 24.90% |
| SEAM — ours, harmful + refusal spaces | 50.55% | 46.42% | 45.88% |
| SEAM — authors’ implementation (reference) | 52.42% | 35.15% | 24.79% |

Capability is the same four-task average (MMLU, TruthfulQA MC1, normalized
ARC-Easy, and normalized HellaSwag), excluding GSM8K. Attacks use the authors’
code for 250 steps. The refusal-corrected variant changes only the refusal
separator from newline to space relative to our original implementation. The
harmful + refusal variant additionally changes the harmful separator to a space;
completion masking, EOS supervision, Alpaca filtering, and optimization settings
remain unchanged. Results use seed 42.

| Harmful | Refusal | Attack | Self-destruct | Before attack | After LR `1e-4` | After LR `2e-4` | W&B |
|---|---|---|---|---:|---:|---:|---|
| `Answer:\n{answer}` | `Answer:\n{refusal}` | `Answer:\n{answer}` | ✗ | 48.42% | 51.40% | 50.13% | [y0iayj61](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/y0iayj61) |
| `Answer:\n{answer}` | `Answer:\n{refusal}` | `Answer: {answer}` | — | 48.42% | — | — | [y0iayj61](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/y0iayj61) |
| `Answer:\n{answer}` | `Answer: {refusal}` | `Answer:\n{answer}` | ✓ | 51.87% | 29.48% | 24.90% | [id2hmrxm](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/id2hmrxm) |
| `Answer:\n{answer}` | `Answer: {refusal}` | `Answer: {answer}` | ✗ / ✓ | 51.87% | 51.23% | 24.82% | [id2hmrxm](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/id2hmrxm) |
| `Answer: {answer}` | `Answer:\n{refusal}` | `Answer:\n{answer}` | — | — | — | — | — |
| `Answer: {answer}` | `Answer:\n{refusal}` | `Answer: {answer}` | — | — | — | — | — |
| `Answer: {answer}` | `Answer: {refusal}` | `Answer:\n{answer}` | ✗ | 50.55% | 46.42% | 45.88% | [l9f421ex](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/l9f421ex) |
| `Answer: {answer}` | `Answer: {refusal}` | `Answer: {answer}` | — | 50.55% | — | — | [l9f421ex](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/l9f421ex) |

Harmful and refusal describe defense training; attack describes attacker training.
Benign formatting stays `Answer: {helpful answer}`. ✓ = substantial capability
collapse; ✗ = no comparable collapse; — = not tried. A single symbol applies to
both tested LRs; `✗ / ✓` means no collapse at `1e-4`, but collapse at `2e-4`. W&B links are defense-training runs. These outcomes refer to our
implementation with seed 42.

Capability uses the same four-task average above. Before-attack scores and W&B
links repeat when attack variants share a defended checkpoint.

## Fresh NPO / SAM 15pp reproduction (2026-11-09)

The fresh run reproduces slower WMDP-Bio recovery with NPO + SAM under the historical 15pp settings. Both defenses were trained from Zephyr-7B-beta for 125 updates, batch size 8, LR 1e-5, and retain coefficient 4. NPO uses beta 0.025; NPO + SAM uses beta 0.05 and rho 0.01. These are independently selected historical settings, not an ablation that changes only SAM.

Both attacks use ordinary cross-entropy for 16 updates, batch size 1, LR 1e-5 with linear decay, AdamW with no weight decay, BF16, sequence length 512, seed 1, and data seed 42. There is no ghost tracking or GIP regularization.

| Method | Step | WMDP-Bio robust ↓ | MMLU-no-bio ↑ |
|---|---:|---:|---:|
| NPO | 0 | 25.35% | 49.01% |
| NPO | 4 | 43.43% | 54.26% |
| NPO | 8 | 47.24% | 54.93% |
| NPO | 12 | 48.27% | 55.33% |
| NPO | 16 | 47.70% | 55.34% |
| NPO + SAM | 0 | 23.16% | 49.57% |
| NPO + SAM | 4 | 25.23% | 45.24% |
| NPO + SAM | 8 | 34.22% | 48.75% |
| NPO + SAM | 12 | 37.21% | 50.87% |
| NPO + SAM | 16 | 37.10% | 51.04% |

At step 16, SAM is 10.60 percentage points lower on WMDP and 4.30 points lower on MMLU. Both models end above their own pre-attack MMLU scores. This is a single-seed reproduction at one attack learning rate.

[Relearning plot (PDF)](../artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--15pp--npo-sam-reproduction.pdf); a PNG and plotted-value CSV are saved beside it. The black line uses the existing October 12 base-model evaluation, not a fresh base-model run.

Manifest: `configs/experiments/z7b-wmdp-bio-npo-sam-repro--15pp.yml`. Submitted source commit: `4ba5748d`. Jobs 680076–680081 all completed with exit code 0:0. Both defense checkpoint-125 files and attack checkpoints 4/8/12/16 were verified as complete model-only saves. All ten full benchmark evaluations use no sample limit and zero shots.

Provenance and raw-result paths: `artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109/`, including `summary.json`, `trajectory.csv`, `completed-jobs.json`, `checkpoint-verification.json`, `figure-verification.json`, exact submitted commands, config hashes, cleanup receipts, and the one-step preflight records. The initially misconfigured preflight 680043 was cancelled and its partial output archived; the corrected preflight 680062 passed before the fresh full runs. Shared 5pp/ghost prepared caches were preserved.
