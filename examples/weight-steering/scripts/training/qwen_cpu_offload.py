"""Support CPU-offloaded Qwen training and text-only FSDP2 resume."""

from dataclasses import replace
from functools import partial

from axolotl.integrations.base import BasePlugin


def all_reduce_offloaded_scalar(value, op, group=None):
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    original_device = value.device
    # FSDP2 CUDA meshes can report backend 'undefined' rather than 'nccl'.
    # Such groups still require CUDA tensors, even when parameters live on CPU.
    if original_device.type == "cpu" and dist.get_backend(group) != "gloo":
        value = value.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(value, op=op, group=group)
    return value.to(original_device)


def restore_qwen_optimizer_state(
    set_state_dict, model, optimizers, state_dict, *, options
):
    """Allow lazy Adam state for the vision tower, requiring all language state."""
    # Full-state loading supplies the checkpoint only on rank zero and broadcasts it.
    if state_dict is not None:
        parameters = {
            name for group in state_dict["param_groups"] for name in group["params"]
        }
        missing = parameters - state_dict["state"].keys()
        unexpected = sorted(
            name for name in missing if not name.startswith("model.visual.")
        )
        if unexpected:
            raise RuntimeError(
                f"Missing language optimizer state in checkpoint: {unexpected}"
            )
        print(
            f"Optimizer resume: {len(missing)} inactive vision parameters "
            "have no saved Adam state.",
            flush=True,
        )
    return set_state_dict(
        model, optimizers, state_dict, options=replace(options, strict=False)
    )


class QwenCPUOffloadPlugin(BasePlugin):
    def post_trainer_create(self, cfg, trainer):
        import axolotl.utils.gradient_clipping as clipping

        clipping._all_reduce_scalar = all_reduce_offloaded_scalar
        if cfg.resume_from_checkpoint:
            import torch.distributed.checkpoint.state_dict as state_dict

            state_dict.set_optimizer_state_dict = partial(
                restore_qwen_optimizer_state, state_dict.set_optimizer_state_dict
            )
