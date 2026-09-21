# FSDP2 Migration Status

## Completed (2026-03-10)

FSDP2 (`fully_shard`) successfully implemented in all four SFT-mode algorithms: `sequential_unlearn_sft.py`, `checkpoint_transfer_unlearn.py`, `lens_unlearn.py`, and `max_update_unlearn.py`.

### Approach

Apply `fully_shard()` manually before creating the HF Trainer, bypassing Trainer/accelerate FSDP config entirely. Override `_prepare_for_training` to skip DDP wrapping (since FSDP2 handles gradient sync).

Key details:
- FSDP2 preserves original parameter shapes as DTensors (unlike FSDP1 which flattens to 1D)
- `param.grad` is available as a DTensor after each backward (unlike FSDP1's FlatParameter)
- FSDP2 model has no `no_sync()` method, so accelerate's `no_sync` falls back to `contextlib.nullcontext` — every backward syncs gradients. This wastes bandwidth on accumulation steps but is correct.
- Mixed precision handled via `torch.amp.autocast` in `training_step` (sequential) or Trainer's built-in autocast (checkpoint transfer, lens, max_update)
- Model saving uses `get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))`

### DTensor compatibility (max_update)

Max update unlearn computes differentiable `||param - w0||` via forward hooks. With FSDP2, params are DTensors, which can't be subtracted from regular tensors. Fix:
- Use `param.full_tensor()` in hooks to extract an autograd-tracked regular tensor from the DTensor
- Use `p.full_tensor()` when cloning initial params to get the complete (unsharded) weight
- DTensor `.sum()` already aggregates across ranks, so manual `dist.all_reduce` is not needed
- Extract remaining DTensor intermediates with `.to_local()` before passing to non-DTensor-aware code

### Verified Jobs

| Job | Algorithm | Result |
|-----|-----------|--------|
| 2692951 | Sequential + AdamW + FSDP2 (bf16) | 32 phases, clean losses, model saved |
| 2692970 | Sequential + Muon + FSDP2 (bf16) | 128 Muon params (2D preserved), clean losses |
| 2693051 | Sequential + FSDP2 (fp16) | Completed, WMDP 42.2%, MMLU 45.7% |
| 2693008 | Checkpoint Transfer + FSDP2 (bf16) | cb_loss 2.07, model saved |
| 2693049 | Checkpoint Transfer + Muon + FSDP2 (bf16) | 128 Muon params, WMDP 35.4%, MMLU 36.4% |
| 2693050 | Checkpoint Transfer + FSDP2 (fp16) | OOM (2 models exceed 95GB in fp16) |
| 2693575 | Lens SFT + FSDP2 (bf16) | forget_loss 2.06, model saved |
| 2693843 | Max Update SFT + FSDP2 (bf16) | sft_loss 2.13, update_norm working, model saved |

### Muon compatibility

Muon now works with sequential and checkpoint transfer unlearn via FSDP2. FSDP1 flattened all params to 1D which made Muon reject them (`ValueError: Muon only supports 2D parameters`). FSDP2 preserves original shapes, so `MuonAdamW` correctly identifies 128 Muon-eligible 2D params and 260 AdamW params.

## Previous Failed Attempt

The earlier FSDP2 attempt (Jobs 2208022, 2208795) used `fsdp_version: 2` in TrainingArguments `fsdp_config`, which told accelerate to use FSDP2. This broke because accelerate's gradient accumulation called `set_requires_gradient_sync(False)`, making `param.grad` None on accumulation steps. Sequential unlearn's two-backward-per-step pattern requires `param.grad` after each backward.

The fix was to apply FSDP2 ourselves (not via accelerate) so that `requires_gradient_sync` stays True always.
