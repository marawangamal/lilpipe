# Gradient Routing (ERA) Experiment Log

**Paper**: *Gradient Routing: Masking Gradients to Localize Computation in Neural Networks* (arXiv 2410.04332)
**Repo**: [kxcloud/gradient-routing](https://github.com/kxcloud/gradient-routing)
**Date**: 2026-02-19 (TinyStories), 2026-02-24 (Virology)

---

## Part 1: TinyStories Forest Unlearning

**Model**: TinyStories-28M (8 layers, 512 d_model, 2048 d_mlp)
**Task**: Unlearn "forest/tree"-related concepts (words: tree, trees, forest, forests, woodland, woodlands)

### Overview

Four configurations of the Expand-Route-Ablate (ERA) framework, varying two axes:
- **Training mode**: from-scratch vs post-training (starting from pretrained TinyStories-28M)
- **Masking scheme**: full_seq (all tokens in forget stories) vs concept (only exact forget word positions)

All runs expand d_mlp by 64 (512→576) across layers 0-4, use DDBP masking, and apply per-dimension learning rates (expanded_lr=1.0, original_lr=-0.75 on target data).

### Experiment Matrix

| ID | Training | Masking | ERA Steps | Coherence Steps | LR | SLURM Job | Status |
|----|----------|---------|-----------|-----------------|------|-----------|--------|
| A | From-scratch | full_seq | 12,500 | -1 (auto) | 5e-4 | 2370544 | Complete |
| B | Post-training | full_seq | 5,000 | 500 | 1e-4 | 2370567 | Complete |
| C | From-scratch | concept | 12,500 | -1 (auto) | 5e-4 | 2370659 | Complete (Phase 3 crashed) |
| D | Post-training | concept | 5,000 | 500 | 1e-4 | 2370672 | Complete (Phase 3 crashed) |

### Summary Table

| Model | Forget Loss | Retain Loss | Target Words | Repetition | Act. Selectivity |
|-------|------------:|------------:|-------------:|-----------:|-----------------:|
| Pretrained baseline | 1.201 | 1.273 | 12 | 0.034 | N/A |
| **A: From-scratch full_seq** | **3.323** | 2.088 | **1** | 0.046 | **2.16x** |
| **B: Post-train full_seq** | 1.337 | **1.122** | 6 | **0.027** | 1.37x |
| C: From-scratch concept | 1.402 | 1.531 | — | — | — |
| D: Post-train concept | 2.222 | 2.238 | — | — | — |

- **Forget Loss** (higher = better unlearning): Model A's forget loss is 2.77x baseline, Model B's is only 1.11x baseline
- **Retain Loss** (lower = better preservation): Model B actually *beats* baseline (1.122 vs 1.273). Model A degrades to 2.088
- **Target Words** in 8 generated samples (lower = better suppression): Model A nearly perfect (1 word), Model B moderate (6, down from 12)
- **Repetition Ratio** (lower = better coherence): Model B best (0.027), Model A comparable to baseline
- **Activation Selectivity** (expanded dim forget/retain ratio, averaged across layers): Model A shows 2.16x selectivity, Model B 1.37x
- **C/D** completed Phases 1-2 but crashed before generation eval (see Phase 3 note below). Validation losses from wandb only.

### Training Losses

| Model | Pre-ablation Forget | Pre-ablation Retain | Post-ablation Forget | Post-ablation Retain |
|-------|--------------------:|--------------------:|---------------------:|---------------------:|
| A: From-scratch full_seq | 1.468 | 1.564 | 3.247 | 2.023 |
| B: Post-train full_seq | 1.086 | 1.079 | 1.346 | 1.105 |

- Model A: ablation increases forget loss by +1.78 (strong routing to expanded dims)
- Model B: ablation increases forget loss by only +0.26 (weak routing — most forget knowledge stays in original dims)

C/D pre/post ablation breakdown not logged separately; only final validation losses available from wandb.

### Coherence Training

| Model | Best Coherence Loss | Step |
|-------|--------------------:|-----:|
| A: From-scratch full_seq | 2.030 | 0 |
| B: Post-train full_seq | 1.081 | 0 |
| C: From-scratch concept | 1.512 | 0 |
| D: Post-train concept | 1.029 | 0 |

All models achieved best coherence at step 0 (coherence fine-tuning did not improve the contracted model).

### Concept Masking Results (C, D)

Both concept masking runs completed Phases 1-2 successfully but crashed during Phase 3 (retraining eval) with the same bug as A/B. Generation evals and activation selectivity were not computed.

**Run C (From-scratch concept):**
- wandb run: cmym4e0s
- Validation forget loss: 1.402, retain loss: 1.531
- Total runtime: 31 min
- Models saved: `concept_results/concept_era_pre_ablation.pt`, `concept_results/concept_era.pt`

**Run D (Post-train concept):**
- wandb run: m6rinjty
- Validation forget loss: 2.222, retain loss: 2.238
- Coherence test loss: 2.251
- Total runtime: 38.5 min
- Models saved: `concept_posttrain_results/concept_posttrain_era_pre_ablation.pt`, `concept_posttrain_results/concept_posttrain_era.pt`

Run D's high forget/retain losses (both ~2.2) suggest the concept masking + post-training combination damaged the model more broadly, unlike full_seq post-training (B) which preserved retain performance.

### Activation Selectivity (Pre-Ablation Models)

Ratio of mean |activation| in expanded dims for forget vs retain data:

**Model A (from-scratch full_seq):**

| Layer | Forget Expanded | Retain Expanded | F/R Ratio |
|-------|----------------:|----------------:|----------:|
| 0 | 0.2623 | 0.0877 | 2.99x |
| 1 | 0.1211 | 0.0344 | 3.53x |
| 2 | 0.0564 | 0.0349 | 1.62x |
| 3 | 0.0600 | 0.0436 | 1.38x |
| 4 | 0.0389 | 0.0309 | 1.26x |

**Model B (post-training full_seq):**

| Layer | Forget Expanded | Retain Expanded | F/R Ratio |
|-------|----------------:|----------------:|----------:|
| 0 | 0.1036 | 0.0566 | 1.83x |
| 1 | 0.1070 | 0.0741 | 1.44x |
| 2 | 0.0996 | 0.0798 | 1.25x |
| 3 | 0.0936 | 0.0750 | 1.25x |
| 4 | 0.0868 | 0.0820 | 1.06x |

Model A shows much stronger selectivity, especially in early layers (3-3.5x). Model B's selectivity is weaker across all layers, consistent with the observation that post-training can't fully segregate forget representations.

### Token Routing Weights

Frequency-based per-token masking (mask_weight=0 → forget/expanded dims, mask_weight=1 → retain/original dims):

**Strongly routed (mask_weight < 0.5):**
| Token | Mask Weight | Forget Freq/10k | Retain Freq/10k |
|-------|------------:|----------------:|----------------:|
| tree | 0.000 | 99.4 | 0.0 |

**Moderately routed (0.5 ≤ mask_weight < 0.85):**
| Token | Mask Weight | Forget Freq/10k | Retain Freq/10k |
|-------|------------:|----------------:|----------------:|
| bird | 0.586 | 73.1 | 18.7 |
| flew | 0.810 | 10.3 | 3.6 |
| bear | 0.816 | 10.9 | 3.8 |

Only 4 of 226 non-rare tokens get meaningful routing. The routing captures semantic associations: `bird`, `flew`, and `bear` co-occur with forest-themed stories even though they're not in the explicit forget list.

### Generation Samples

**Prompt: "Once upon a time, Timmy went to the forest"**

| Model | Target Words | Sample |
|-------|:------------:|--------|
| Pretrained | 4 | "...to find a new adventure. He walked and walked until he found a big rock..." |
| A: From-scratch | 1 | "...to no there was a not to what to go to ask, there was a no to that it was a not to play..." |
| B: Post-train | 1 | "...to pick berries. All of his berries were bright and red and look delicious..." |

**Prompt: "The little bear lived in the"**

| Model | Target Words | Sample |
|-------|:------------:|--------|
| Pretrained | 3 | "...forest. He had a big, furry tail and he loved to play with it..." |
| A: From-scratch | 0 | "...just for a while he was there was on he could not a his new made a mom..." |
| B: Post-train | 1 | "...forest. He was three years old and very excited to explore..." |

**Prompt: "Once upon a time, there was a little girl named Lily"**

| Model | Target Words | Sample |
|-------|:------------:|--------|
| Pretrained | 0 | "...She loved to eat spaghetti with a fork..." (coherent, on-topic) |
| A: From-scratch | 0 | "...She loved to play outside with her friends..." (mostly coherent) |
| B: Post-train | 0 | "...She loved to play outside in the sun..." (fully coherent) |

Key pattern: Model A generates incoherent text when the prompt activates forest-related contexts (the forget knowledge is removed but nothing fills the gap). On neutral prompts, it's closer to normal. Model B generates fluent text regardless but doesn't fully suppress forest content.

### Phase 3: Retraining Resistance

Retrain on 64 forget stories for up to 40 steps (early stop after 3 consecutive forget loss increases). 5 retrain trials averaged. Eval on held-out forget/retain validation sets (5,800 forget, 21,716 retain — full validation split). SLURM job 2455825.

**Retraining curves (averaged across 5 trials):**

**Model A (From-scratch full_seq):**

| Step | Forget Loss | Retain Loss |
|-----:|------------:|------------:|
| 0 | 3.330 | 2.055 |
| 5 | 1.935 | 1.684 |
| 10 | 1.749 | 1.675 |
| 15 | 1.673 | 1.665 |
| 20 | 1.643 | 1.670 |
| 25 | 1.640 | 1.689 |
| 31 | 1.640 | 1.703 |

Early stopped at step ~30. Forget loss drops 3.33→1.64 (51% reduction). Slow convergence — the unlearned knowledge is partially recoverable but requires many steps and causes retain degradation.

**Model B (Post-train full_seq):**

| Step | Forget Loss | Retain Loss |
|-----:|------------:|------------:|
| 0 | 1.361 | 1.107 |
| 1 | 1.161 | 1.112 |
| 2 | 1.146 | 1.120 |
| 3 | 1.137 | 1.127 |
| 4 | 1.140 | 1.137 |
| 5 | 1.151 | 1.151 |
| 6 | 1.167 | 1.169 |

Early stopped at step 6. Near-instant recovery: forget loss drops 1.36→1.14 in 3 steps. No retraining resistance.

**Model C (From-scratch concept):**

| Step | Forget Loss | Retain Loss |
|-----:|------------:|------------:|
| 0 | 1.407 | 1.527 |
| 1 | 1.403 | 1.526 |
| 2 | 1.406 | 1.532 |
| 3 | 1.411 | 1.537 |
| 4 | 1.416 | 1.542 |

Early stopped at step 4. Forget loss barely changed (1.407→1.403→1.416) — the model's forget loss was already low (weak unlearning) so there's nothing to retrain back.

**Model D (Post-train concept):**

| Step | Forget Loss | Retain Loss |
|-----:|------------:|------------:|
| 0 | 0.994 | 1.061 |
| 1 | 1.007 | 1.078 |
| 2 | 1.015 | 1.088 |
| 3 | 1.036 | 1.109 |
| 4 | 1.065 | 1.136 |
| 5 | 1.092 | 1.160 |

Early stopped at step 4-5. Forget loss *increases* from step 0 — the model overfits on the small forget set, worsening both forget and retain loss. No meaningful knowledge recovery.

---

## Part 2: Virology ERA (WMDP-Bio Unlearning)

**Model**: EleutherAI/deep-ignorance-unfiltered (32 layers, d_model=4096, d_mlp=16384)
**Task**: Unlearn WMDP-Bio hazardous knowledge
**Date**: 2026-02-24

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | EleutherAI/deep-ignorance-unfiltered |
| Forget dataset | Unlearning/WMDP-Bio-Remove-Dataset |
| Retain dataset | EleutherAI/wikitext_document_level |
| d_mlp expansion | 128 (16384→16512, 0.78%) |
| Layers masked | 0-7 (first 8 of 32) |
| ERA training steps | 2,000 |
| Coherence finetuning steps | 500 |
| Batch size | 2 (grad_accum=8, effective=16) |
| Sequence length | 512 tokens |
| Learning rate | 5e-5 (cosine decay to 5e-6) |
| Masking | full_seq, DDBP |
| Dim LRs | expanded_target=1.0, original_target=-0.75, off-target=1.0 |
| SLURM Job | 2450720 |

### Main Run Results

**Training Losses:**

| Phase | Forget Loss | Retain Loss |
|-------|------------:|------------:|
| Step 0 | 1.964 | 2.013 |
| Step 1000 | 1.891 | 1.717 |
| Step 1750 (final) | 1.886 | 1.701 |
| Pre-ablation | 1.888 | 1.738 |
| Post-ablation | 1.888 | 1.738 |
| Best coherence | 1.774 (step 0) | — |

Ablation has negligible effect: forget loss changes by -0.0003, retain loss by -0.0005. The expanded dimensions are not being used preferentially for forget data.

**Benchmark Evaluation:**

| Benchmark | Unlearned | Baseline | Delta |
|-----------|----------:|---------:|------:|
| **WMDP Bio Robust (overall)** | 0.4412 ± 0.017 | 0.4297 ± 0.017 | +0.012 |
| Bioweapons & bioterrorism | 0.3737 ± 0.035 | 0.3526 ± 0.035 | +0.021 |
| Dual use virology | 0.2857 ± 0.087 | 0.2857 ± 0.087 | 0.000 |
| Enhanced pandemic pathogens | 0.4118 ± 0.049 | 0.3725 ± 0.048 | +0.039 |
| Expanding access to threats | 0.3810 ± 0.109 | 0.3810 ± 0.109 | 0.000 |
| Reverse genetics & editing | 0.5000 ± 0.037 | 0.5054 ± 0.037 | -0.005 |
| Viral vector research | 0.4721 ± 0.027 | 0.4633 ± 0.027 | +0.009 |
| **MMLU (overall)** | 0.4568 ± 0.004 | 0.4576 ± 0.004 | -0.001 |
| MMLU - humanities | 0.4268 ± 0.007 | 0.4236 ± 0.007 | +0.003 |
| MMLU - other | 0.5208 ± 0.009 | 0.5294 ± 0.009 | -0.009 |
| MMLU - social sciences | 0.5288 ± 0.009 | 0.5294 ± 0.009 | -0.001 |
| MMLU - stem | 0.3685 ± 0.008 | 0.3673 ± 0.008 | +0.001 |

ERA produced no meaningful unlearning. WMDP accuracy slightly *increased* (0.4412 vs 0.4297), and MMLU is essentially unchanged. All deltas are within standard error.

### Hyperparameter Sweep

Swept d_mlp expansion size and original_dim_lr across 8 layers. All runs use 2000 ERA steps, batch_size=8, truncate_at=1024.

**Completed sweep runs (8 layers):**

| d_mlp | orig_lr | Pre-Forget | Pre-Retain | Post-Forget | Post-Retain | ΔForget | ΔRetain |
|------:|--------:|-----------:|-----------:|------------:|------------:|--------:|--------:|
| 256 | -0.50 | 1.886 | 1.737 | 1.886 | 1.737 | -0.000 | -0.000 |
| 256 | -1.00 | 1.925 | 1.754 | 1.912 | 1.750 | -0.013 | -0.005 |
| 256 | -1.50 | 2.205 | 1.886 | 2.096 | 1.828 | -0.109 | -0.057 |
| 512 | -0.50 | 1.886 | 1.738 | 1.886 | 1.737 | -0.000 | -0.001 |
| 512 | -1.00 | 1.916 | 1.751 | 1.903 | 1.746 | -0.013 | -0.005 |
| 512 | -1.50 | 2.582 | 2.170 | 2.284 | 2.012 | -0.298 | -0.158 |
| 1024 | -0.50 | 1.887 | 1.739 | 1.886 | 1.738 | -0.001 | -0.002 |
| 1024 | -1.00 | 1.933 | 1.758 | 1.901 | 1.747 | -0.032 | -0.011 |
| 1024 | -1.50 | 3.044 | 2.621 | 3.103 | 3.122 | +0.059 | +0.501 |
| 2048 | -0.50 | 1.888 | 1.741 | 1.887 | 1.739 | -0.001 | -0.002 |
| 2048 | -1.00 | 1.923 | 1.754 | 1.895 | 1.744 | -0.028 | -0.010 |
| 2048 | -1.50 | 2.114 | 1.821 | 1.918 | 1.760 | -0.195 | -0.061 |
| 4096 | -0.50 | 1.892 | 1.745 | 1.887 | 1.741 | -0.005 | -0.004 |
| 4096 | -1.00 | 1.951 | 1.767 | 1.894 | 1.746 | -0.057 | -0.021 |
| 4096 | -1.50 | 2.049 | 1.813 | 1.910 | 1.756 | -0.139 | -0.057 |
| 4096 | -2.00 | 3.590 | 3.537 | 9.356 | 8.962 | +5.766 | +5.425 |

**Cancelled sweep runs (16L/32L at lr=-0.75):** 16 jobs testing layer counts of 8/16/32 were cancelled before completion (SLURM batch 2450951-2450966). Only config.json files saved, no results.

**Sweep observations:**
- orig_lr=-0.50: No meaningful ablation effect at any expansion size
- orig_lr=-1.00: Small but consistent post-ablation forget reduction (~0.01-0.06)
- orig_lr=-1.50: Larger effects but unstable — 1024×-1.50 shows retain loss *increasing* post-ablation (+0.50)
- orig_lr=-2.00: Catastrophic model collapse (only tested at 4096)
- Expansion size has minimal impact: 256 and 4096 achieve similar results at same LR
- No sweep configuration was evaluated on WMDP/MMLU benchmarks

### Notes

- All runs use `WANDB_MODE=disabled` for TinyStories, enabled for Virology
- System: NVIDIA GH200 120GB, aarch64, CUDA 12.7, torch 2.10.0+cu126
- TinyStories env: `gradient-routing` (Python 3.11, TransformerLens from commit a52bfac)
- Virology env: same `gradient-routing` conda env
- Model storage: `~/team-shard-filesystem/models/` (TinyStories), `projects/virology_era/outputs/` (Virology)
- Eval script: `projects/tinystories/eval_compare.py` (TinyStories), `projects/virology_era/evaluate_virology.py` (Virology)
- Full TinyStories eval results: `eval_output/eval_comparison.json`
- Full Virology eval results: `projects/virology_era/outputs/full_run/eval_results.json`
- Virology sweep results: `projects/virology_era/outputs/sweep/exp*/results.json`

---

## Part 3: Paper Reproduction (TinyStories ERAC)

**Date**: 2026-02-25
**Goal**: Reproduce Section 4.2.1 of arXiv 2410.04332 with the paper's exact config
**SLURM Job**: 2474717 (10 seeds × 5 baselines), 2475944 (retrain LR sweep)

### Config Differences from Part 1

Part 1 experiments used a debug config from `tinystories_era.py __main__`. Key differences:

| Parameter | Part 1 (Model A) | Paper Config |
|-----------|------------------|--------------|
| ERA steps | 12,500 | 5,000 |
| Coherence steps | -1 (auto) | 40 |
| Validation stories | 100 | 3,000 |
| Seeds | 1 | 10 |
| Retraining stories | [64] | [1, 4, 16, 64] |
| Baselines | ERAC only | ERAC, demix, base, pure, expanded_base |

Paper config source: `bulk_runs_for_paper.py`. Note: that file has a bug on line 166 — passes `demix_cfg` to all configs including ERAC. Our `run_reproduction.py` fixes this.

### ERAC Results (4 of 10 seeds complete)

Phase 3 retraining uses: lr=5e-5, AdamW(betas=0.9,0.95), 40 steps max, eval every step, early stop after 3 consecutive forget loss increases. Batch size = num_stories (1, 4, 16, or 64 stories fed as one batch).

**Mean over seeds 42, 1337, 2718, 31415:**

| ns | pre_forget | pre_retain | post_forget | post_retain | retrain_forget | retrain_retain |
|---:|-----------:|-----------:|------------:|------------:|---------------:|---------------:|
| 1 | 1.5169 | 1.6220 | 1.9178 | 1.6735 | 1.7015 | 1.6661 |
| 4 | 1.5169 | 1.6220 | 1.9178 | 1.6735 | 1.6522 | 1.6674 |
| 16 | 1.5169 | 1.6220 | 1.9178 | 1.6735 | 1.6118 | 1.6730 |
| 64 | 1.5169 | 1.6220 | 1.9178 | 1.6735 | 1.5720 | 1.6721 |

Columns: pre_forget/pre_retain = before ablation (fully trained model). post_forget/post_retain = after ablation. retrain_forget/retrain_retain = at step of minimum forget loss during retraining attack.

**Per-seed results:**

seed 42 — post-ablation delta: forget +0.340, retain +0.038

| ns | pre_forget | pre_retain | post_forget | post_retain | retrain_forget | retrain_retain |
|---:|-----------:|-----------:|------------:|------------:|---------------:|---------------:|
| 1 | 1.5126 | 1.6116 | 1.8531 | 1.6494 | 1.6794 | 1.6520 |
| 4 | 1.5126 | 1.6116 | 1.8531 | 1.6494 | 1.6270 | 1.6511 |
| 16 | 1.5126 | 1.6116 | 1.8531 | 1.6494 | 1.5969 | 1.6536 |
| 64 | 1.5126 | 1.6116 | 1.8531 | 1.6494 | 1.5679 | 1.6538 |

seed 1337 — post-ablation delta: forget +0.632, retain +0.072

| ns | pre_forget | pre_retain | post_forget | post_retain | retrain_forget | retrain_retain |
|---:|-----------:|-----------:|------------:|------------:|---------------:|---------------:|
| 1 | 1.5406 | 1.6429 | 2.1729 | 1.7148 | 1.7741 | 1.6911 |
| 4 | 1.5406 | 1.6429 | 2.1729 | 1.7148 | 1.7190 | 1.7046 |
| 16 | 1.5406 | 1.6429 | 2.1729 | 1.7148 | 1.6621 | 1.7056 |
| 64 | 1.5406 | 1.6429 | 2.1729 | 1.7148 | 1.6109 | 1.7020 |

seed 2718 — post-ablation delta: forget +0.319, retain +0.060

| ns | pre_forget | pre_retain | post_forget | post_retain | retrain_forget | retrain_retain |
|---:|-----------:|-----------:|------------:|------------:|---------------:|---------------:|
| 1 | 1.4955 | 1.6043 | 1.8143 | 1.6639 | 1.6418 | 1.6490 |
| 4 | 1.4955 | 1.6043 | 1.8143 | 1.6639 | 1.6011 | 1.6462 |
| 16 | 1.4955 | 1.6043 | 1.8143 | 1.6639 | 1.5682 | 1.6545 |
| 64 | 1.4955 | 1.6043 | 1.8143 | 1.6639 | 1.5392 | 1.6475 |

seed 31415 — post-ablation delta: forget +0.319, retain +0.037

| ns | pre_forget | pre_retain | post_forget | post_retain | retrain_forget | retrain_retain |
|---:|-----------:|-----------:|------------:|------------:|---------------:|---------------:|
| 1 | 1.5189 | 1.6292 | 1.8308 | 1.6660 | 1.7106 | 1.6721 |
| 4 | 1.5189 | 1.6292 | 1.8308 | 1.6660 | 1.6615 | 1.6676 |
| 16 | 1.5189 | 1.6292 | 1.8308 | 1.6660 | 1.6198 | 1.6782 |
| 64 | 1.5189 | 1.6292 | 1.8308 | 1.6660 | 1.5702 | 1.6848 |

### Pending

- [ ] Seeds 4-9 ERAC + all seeds for demix, base, pure, expanded_base (SLURM 2474717 running)
- [ ] Retrain LR sweep: lr={1e-5, 3e-5, 5e-5}, batch sizes {1,2,4,8,16,64} (SLURM 2475944)
- [ ] Virology paper reproduction (paper uses 0.7B model, 20-token routing, FineWeb-Edu — completely different from Part 2's 6.85B config)
- [ ] Run `analyze_bulk_runs.py` on completed reproduction data
- [x] ~~Fix Phase 3 retraining resistance bug~~ (fixed in Part 1)
- [ ] Run generation eval for concept masking models C/D (weights saved)
