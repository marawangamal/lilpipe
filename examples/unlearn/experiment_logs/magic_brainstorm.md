# MAGIC Attribution for Tamper-Resistant Unlearning: Brainstorming

## Problem Statement

Achieve tamper-resistant unlearning of WMDP-bio-robust knowledge while maintaining MMLU.

Current best results across all methods:

| Method | Dataset | WMDP Robust | MMLU | Tamper Resistant? |
|--------|---------|-------------|------|-------------------|
| Baseline | — | 42.97% | 45.10% | N/A |
| Tuned Lens (LoRA r=16, 8k ex, 256 steps) | WMDP-bio-forget | 23.16% | 43.04% | Untested |
| Sequential SFT + keyword + L2SP | WMDP-bio-forget + WikiText | 28.95% | 44.67% | No (recovers step 10) |
| Orth CB SFT (rm15, orth15, ret15, 512 steps) | WMDP-bio-forget + WikiText | 28.11% | 43.53% | No (for r=16 LoRA) |
| Orth CB SFT (rm15, orth15, ret50, 512 steps) | WMDP-bio-forget + WikiText | 33.64% | 44.81% | No (for r=16 LoRA) |
| Filtered model (e2e-strong-filter) | Pretraining filter | 34.56% | 46.00% | Partially (stays 33-36%) |
| Orth CB LoRA r=8 all-module (pdbs=1, ret=0) | WMDP-bio-forget + WikiText | 26.2% | 25.5% | False positive (model destroyed) |

No method achieves all three: low WMDP + high MMLU + tamper resistance.

## What MAGIC Gives Us

MAGIC per-token attribution scores tell us: for each token in the training data, how much does training on that token increase/decrease the eval loss on a specific benchmark (WMDP or MMLU).

We now have two sets of scores on the same 10k WMDP-bio-forget-corpus tokens:
- **WMDP scores**: per-token contribution to WMDP-bio-robust eval loss
- **MMLU scores**: per-token contribution to MMLU eval loss

This gives us a 2D map of each token's effect on both benchmarks simultaneously.

## Previous Attribution Experiments

Six experiments in `experiment_logs/attribution_unlearning.md`:

| Experiment | MAGIC Training Data | MAGIC Eval | Fine-tune Data | Method |
|---|---|---|---|---|
| 1-2 (weighted/sign) | bio-retain (1k) | WMDP | bio-retain (1k) | Attribution-weighted loss (GA on negative, SFT on positive) |
| 3 (positive only) | WikiText (1k) | WMDP | WikiText (1k) | SFT on WMDP+ tokens only |
| 4 (negative only) | WikiText (1k) | WMDP | WikiText (1k) | SFT on WMDP- tokens only |
| 5 (selective) | WikiText (1k) | WMDP + MMLU | WikiText (1k) | SFT on WMDP+ AND MMLU<=0 tokens (5,538 / 103,807 = 5.3%) |
| 6 (compatible) | WikiText (1k) | WMDP + MMLU | WikiText (1k) | SFT on WMDP+ AND MMLU- tokens |

All experiments showed the same pattern: WMDP drops to ~24% only at learning rates that also collapse MMLU to ~25%.

## Approaches to Explore

### Approach 1: Attribution-Guided Circuit Breaking

Use MAGIC scores to improve orth circuit breakers.

**Idea**: Instead of applying circuit breaker loss uniformly to all tokens in forget data, weight the cb_loss by WMDP attribution magnitude. Tokens with high WMDP attribution get stronger circuit breaking; tokens that also help MMLU get weaker breaking.

**Why this might help**: Current orth CB applies equal pressure to all forget tokens. Some of those tokens encode general language patterns that MMLU also needs. Selectively pressuring only the WMDP-specific tokens could preserve MMLU better while still achieving strong unlearning.

**Implementation**: Modify `orth_circuit_breakers.py` to accept per-token weights from MAGIC scores. Weight the cb_loss computation by normalized WMDP attribution scores.

### Approach 2: Attribution-Selected Forget Corpus

Use MAGIC scores to select the most impactful subset of the forget corpus for unlearning.

**Idea**: Rather than unlearning on the full forget corpus, select the top-k examples/tokens by WMDP attribution score (and filter out those with high MMLU attribution). This concentrates the unlearning signal on the tokens that matter most.

**Why this might help**: Current methods use the full forget corpus which includes many tokens that encode general knowledge. By selecting only high-WMDP, low-MMLU tokens, we reduce collateral damage. This is similar to the "selective" experiment but on bio-specific data where the separation should be much cleaner.

**Implementation**: Score the 10k forget examples, select top 1-2k by (WMDP_score - MMLU_score), use those as the forget set for any unlearning method.

### Approach 3: Two-Phase Unlearning (Targeted then Stabilized)

Phase 1: Attribution-guided aggressive unlearning on high-WMDP tokens.
Phase 2: Stabilization training on high-MMLU tokens to restore capabilities.

**Why this might help**: Disentangles the unlearning and retention objectives. Phase 1 can be aggressive without worrying about MMLU. Phase 2 can focus solely on MMLU recovery without re-teaching WMDP knowledge (by only training on MMLU-attributed tokens).

### Approach 4: Attribution-Guided Representation Engineering

Use MAGIC scores to identify which internal representations encode WMDP knowledge.

**Idea**:
1. Compute MAGIC attribution scores (done)
2. For high-WMDP tokens, extract hidden state representations at each layer
3. Identify the representation directions that are specific to WMDP (high WMDP attribution) but not MMLU
4. Apply targeted representation editing to remove those directions

**Why this might help**: Instead of modifying outputs (like circuit breakers), we modify the internal representation space. If we can identify and remove the specific directions that encode bio-hazard knowledge, the removal should be more persistent under finetuning.

**Relation to orth loss**: This is conceptually similar to what orth_loss tries to do (diversify forget representations), but guided by actual attribution data rather than a general diversity prior.

### Approach 5: Attribution-Weighted Gradient Ascent with Orth Regularization

Combine the attribution weighting with orthogonal regularization.

**Idea**:
1. Gradient ascent on forget tokens, weighted by WMDP attribution scores
2. Simultaneously apply orth_loss to ensure forget representations are diverse
3. Retain loss on high-MMLU tokens only (not all retain data)

**Key insight**: The orth CB experiments showed that pdbs>=4 (with active orth loss) creates more persistent unlearning. Adding attribution-based token weighting on top should make the unlearning more targeted. The retain loss guided by MMLU attribution should better preserve capabilities.

### Approach 6: Attribution-Based Curriculum for Unlearning

Order the unlearning curriculum by attribution score magnitude.

**Idea**: Start by unlearning the highest-WMDP, lowest-MMLU tokens first (easiest, most targeted). Gradually move to tokens with more mixed attributions. This gives the model time to restructure its representations before tackling harder cases.

**Why**: Current methods apply all forget data at once, forcing the model to simultaneously handle easy (purely bio) and hard (bio + general) tokens. A curriculum approach could be more stable.

## Key Questions the 10k Forget Attribution Will Answer

1. **How separated are WMDP and MMLU attributions on forget data?** If most tokens have high WMDP and low MMLU attribution, selective unlearning is feasible. If attributions are highly correlated (as they were on WikiText), we need a different approach.

2. **What fraction of forget tokens are "purely WMDP"?** Tokens with WMDP_score >> 0 and MMLU_score <= 0. This determines how much surgical precision is possible.

3. **Are there examples that are entirely WMDP-specific?** If some examples have high aggregate WMDP score but near-zero MMLU score, we can use example-level selection in addition to token-level.

4. **What's the magnitude ratio?** If WMDP attributions on forget data are 10x stronger than MMLU attributions, even imprecise targeting should work.

## Experimental Plan

### Phase 1: Analyze Attribution Scores (Pending Jobs 2451730, 2451731)
- Compare WMDP vs MMLU attribution distributions on forget corpus
- Compute correlation between WMDP and MMLU token scores
- Identify the separable fraction (high WMDP, low MMLU)
- Visualize per-example WMDP vs MMLU aggregate scores

### Phase 2: Selective Unlearning on Forget Corpus
- Use existing `validate_attribution.py` framework with forget corpus scores
- Experiment with different selection criteria:
  a. WMDP-positive tokens only
  b. WMDP-positive AND MMLU-nonpositive tokens
  c. Top-k by (WMDP_score - alpha * MMLU_score) for various alpha
- Evaluate WMDP-bio-robust and MMLU at each setting

### Phase 3: Integration with Orth Circuit Breakers
- Modify orth_circuit_breakers.py to accept per-token weights
- Run standard orth CB training but with attribution-weighted cb_loss
- Test tamper resistance on the resulting models

### Phase 4: Tamper Resistance Testing
- Run standard tamper attacks (512 examples, lr=2e-5, 30 epochs) on best models
- Compare against:
  a. Standard orth CB (unweighted)
  b. Filtered model baseline
  c. Best previous non-resistant methods (tuned lens, sequential SFT)

## Notes on Tamper Resistance Mechanisms

From the orth CB experiments, tamper resistance seems to require:

1. **All-layer coverage**: Excluding any layer allows recovery through that layer
2. **Representation diversity** (orth loss): Prevents single-direction recovery
3. **Sufficient weight modification**: Low-rank updates (r=2,4) easily reversed; higher rank (r>=8) more persistent
4. **Not just output masking**: Methods that only affect output layer are trivially reversed

MAGIC attribution might help with a fundamentally different mechanism: if we can identify and surgically remove the specific knowledge from the weights (rather than masking/redirecting it), the knowledge simply isn't there to recover. The question is whether MAGIC scores are precise enough to guide this removal.

## Integration Points in Orth Circuit Breakers

The cb_loss in orth_circuit_breakers.py is already computed per-token:

```python
# Per-token cosine similarity: [B, S]
cos_sim = (norm_lora * norm_ref).sum(dim=-1)

# Currently uniformly weighted by attention mask:
cos_sim = cos_sim * circuit_breaker_attention_mask  # [B, S]
cb_loss_total = cb_loss_total + torch.relu(cos_sim).sum()
```

Attribution weights can be injected by replacing the uniform attention mask with weighted attribution scores:

```python
# With attribution weights:
attribution_weights = inputs.get("bio_remove_attribution_weights")  # [B, S]
if attribution_weights is not None:
    cos_sim = cos_sim * attribution_weights.to(target_device)
else:
    cos_sim = cos_sim * circuit_breaker_attention_mask
```

The existing keyword_mask infrastructure (keyword_masks.py) already handles per-token binary masks via the dataset pipeline. Attribution weights can follow the same pattern, extended to float weights.

Orth loss also has an integration point: the sequence pooling step (mean over tokens) can be weighted by attribution scores to focus the orthogonality constraint on high-attribution tokens.

## Risk: Attribution Scores May Not Transfer to Unlearning

MAGIC scores are computed by backpropagating through SGD training on the forget corpus. They tell us which tokens contributed to learning the knowledge during that specific 250-step SGD trajectory. But:

1. The base model already has this knowledge (it was trained on much more data)
2. Unlearning needs to modify the existing representations, not just prevent learning
3. The attribution scores describe "learning dynamics" not "knowledge location"

Counter-argument: if a token has high WMDP attribution, it means training on that token significantly changes the model's WMDP accuracy. This suggests the token activates the relevant knowledge circuits. Gradient ascent on those specific tokens should reverse that activation.
