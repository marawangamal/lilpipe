"""Retain fine-tuning with weight distance maximized."""

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open

try:
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()
    distribute_tensor = None


def global_rms_distance(parameters, anchors, eps=1e-8):
    """Return the authors' normalized full-model RMS parameter distance."""
    pairs = tuple(zip(parameters, anchors, strict=True))
    if not pairs:
        raise ValueError("global_rms_distance requires at least one parameter")

    uses_dtensor = any(isinstance(parameter, DTensor) for parameter, _ in pairs)
    if uses_dtensor and not dist.is_initialized():
        raise RuntimeError(
            "DTensor weight distance requires distributed initialization"
        )

    device = pairs[0][0].device
    value_squared = torch.zeros((), device=device, dtype=torch.float32)
    gradient_squared = torch.zeros((), device=device, dtype=torch.float32)
    element_count = sum(parameter.numel() for parameter, _ in pairs)
    world_size = dist.get_world_size() if uses_dtensor else 1

    for parameter, anchor in pairs:
        if isinstance(parameter, DTensor):
            delta = parameter.to_local().float() - anchor.to_local().float()
            is_replicated = all(
                isinstance(placement, Replicate) for placement in parameter.placements
            )
            value_divisor = world_size if is_replicated else 1
        else:
            delta = parameter.float() - anchor.float()
            value_divisor = 1
        squared = delta.square().sum()
        value_squared = value_squared + squared / value_divisor
        gradient_squared = gradient_squared + squared

    global_squared = value_squared.detach().clone()
    if uses_dtensor:
        dist.all_reduce(global_squared, op=dist.ReduceOp.SUM)
    distance = (global_squared / element_count + eps).sqrt()
    local_gradient = (gradient_squared - gradient_squared.detach()) / (
        2 * distance * element_count
    )
    return distance + local_gradient


def load_reference_parameters(parameters, model_name_or_path):
    """Load reference weights and shard them like the corresponding parameters."""
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(repo_id=model_name_or_path, local_files_only=True)
        )

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]

    checkpoint_names = {
        name: name.replace("._checkpoint_wrapped_module", "").replace(
            "._fsdp_wrapped_module", ""
        )
        for name in parameters
    }
    missing = {
        checkpoint_name
        for checkpoint_name in checkpoint_names.values()
        if checkpoint_name not in weight_map
    }
    if missing:
        names = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"anchor checkpoint is missing parameters: {names}")

    names_by_shard = defaultdict(list)
    for name, checkpoint_name in checkpoint_names.items():
        names_by_shard[weight_map[checkpoint_name]].append(name)

    reference_parameters = {}
    for shard_name, names in names_by_shard.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                parameter = parameters[name]
                reference_parameter = shard.get_tensor(checkpoint_names[name]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if isinstance(parameter, DTensor):
                    reference_parameter = distribute_tensor(
                        reference_parameter,
                        parameter.device_mesh,
                        parameter.placements,
                        src_data_rank=None,
                    )
                reference_parameters[name] = reference_parameter.detach()
    return reference_parameters


class WeightDistanceMixin:
    """Provide an RMS weight-distance loss against fixed reference weights."""

    def __init__(
        self,
        *args,
        reference_model_name_or_path,
        coefficient_weight_dist=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.coefficient_weight_dist = coefficient_weight_dist
        self.reference_model_name_or_path = reference_model_name_or_path
        self._theta_ref = None

    @property
    def theta(self):
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    @property
    def theta_ref(self):
        theta = self.theta
        if self._theta_ref is None:
            self._theta_ref = load_reference_parameters(
                theta, self.reference_model_name_or_path
            )
        if theta.keys() != self._theta_ref.keys():
            raise RuntimeError(
                "trainable parameters changed after weight-distance snapshot"
            )
        return self._theta_ref

    def compute_weight_dist_loss(self, return_grads=False):
        loss_weight_dist = -self.coefficient_weight_dist * global_rms_distance(
            self.theta.values(), self.theta_ref.values()
        )
        if return_grads:
            grad_weight_dist = torch.autograd.grad(
                loss_weight_dist,
                self.theta.values(),
                retain_graph=True,
                allow_unused=True,
            )
            return loss_weight_dist, grad_weight_dist
        return loss_weight_dist
