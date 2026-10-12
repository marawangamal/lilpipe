"""Gradient norms shared by gradient-difference trainers."""

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()


def global_l2_norm(tensors):
    """Return the full-model norm for ordinary or FSDP2-sharded tensors."""
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("global_l2_norm requires at least one tensor")

    uses_dtensor = any(isinstance(tensor, DTensor) for tensor in tensors)
    world_size = dist.get_world_size() if uses_dtensor else 1
    norm_device = tensors[0].device
    local_squared_norm = torch.zeros((), device=norm_device, dtype=torch.float32)
    for tensor in tensors:
        if isinstance(tensor, DTensor):
            local_tensor = tensor.to_local()
            is_replicated = all(
                isinstance(placement, Replicate) for placement in tensor.placements
            )
            divisor = world_size if is_replicated else 1
        else:
            local_tensor = tensor
            divisor = 1
        squared_norm = local_tensor.detach().float().square().sum() / divisor
        local_squared_norm += squared_norm.to(norm_device)

    if uses_dtensor:
        if not dist.is_initialized():
            raise RuntimeError("DTensor gradients require distributed initialization")
        dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    return local_squared_norm.sqrt()


def gradient_dict_l2_norm(gradients):
    """Return the norm of the non-None gradients in a parameter mapping."""
    active = [gradient for gradient in gradients.values() if gradient is not None]
    if not active:
        raise RuntimeError("loss term produced no parameter gradients")
    return global_l2_norm(active)
