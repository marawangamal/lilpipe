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

from axolotl.core.trainers.base import AxolotlTrainer


def global_l2_distance(parameters, anchors):
    """Return a differentiable full-model Euclidean distance."""
    pairs = tuple(zip(parameters, anchors, strict=True))
    if not pairs:
        raise ValueError("global_l2_distance requires at least one parameter")

    uses_dtensor = any(isinstance(parameter, DTensor) for parameter, _ in pairs)
    if uses_dtensor and not dist.is_initialized():
        raise RuntimeError(
            "DTensor weight distance requires distributed initialization"
        )

    device = pairs[0][0].device
    value_squared = torch.zeros((), device=device, dtype=torch.float32)
    gradient_squared = torch.zeros((), device=device, dtype=torch.float32)
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
    if global_squared == 0:
        return gradient_squared * 0

    distance = global_squared.sqrt()
    # Preserve the global value while supplying the correct local derivative.
    return distance + (gradient_squared - gradient_squared.detach()) / (2 * distance)


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


def load_pretrained_anchors(parameters, model_name_or_path):
    """Load anchor weights and shard them like the corresponding parameters."""
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

    anchors = {}
    for shard_name, names in names_by_shard.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                parameter = parameters[name]
                anchor = shard.get_tensor(checkpoint_names[name]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if isinstance(parameter, DTensor):
                    anchor = distribute_tensor(
                        anchor,
                        parameter.device_mesh,
                        parameter.placements,
                        src_data_rank=None,
                    )
                anchors[name] = anchor.detach()
    return anchors


class WeightDistTrainer(AxolotlTrainer):
    """Minimize retain loss while maximizing distance from initial weights."""

    distance_coefficient = 1.0
    distance_function = staticmethod(global_l2_distance)

    def __init__(self, *args, distance_coefficient=None, **kwargs):
        super().__init__(*args, **kwargs)
        if distance_coefficient is not None:
            self.distance_coefficient = distance_coefficient
        self._weight_dist_anchors = None
        self._weight_dist_logged_step = None

    def _parameters_and_anchors(self, model):
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if self._weight_dist_anchors is None:
            self._weight_dist_anchors = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
        if parameters.keys() != self._weight_dist_anchors.keys():
            raise RuntimeError(
                "trainable parameters changed after weight-distance snapshot"
            )
        return parameters, self._weight_dist_anchors

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        parameters, anchors = self._parameters_and_anchors(model)
        fields = ("input_ids", "attention_mask", "labels")
        outputs = model(**{field: inputs[field] for field in fields})
        retain_loss = outputs.loss.float()
        distance = self.distance_function(parameters.values(), anchors.values())
        step = self.state.global_step
        if step != self._weight_dist_logged_step:
            self.log({"weight_distance": distance.detach().item()})
            self._weight_dist_logged_step = step
        loss = retain_loss - self.distance_coefficient * distance
        return (loss, outputs) if return_outputs else loss


class PretrainedAnchorWeightDistTrainer(WeightDistTrainer):
    """Maximize distance from Zephyr while fine-tuning on retain data."""

    anchor_model_name_or_path = "HuggingFaceH4/zephyr-7b-beta"
    distance_function = staticmethod(global_rms_distance)

    def _parameters_and_anchors(self, model):
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if self._weight_dist_anchors is None:
            self._weight_dist_anchors = load_pretrained_anchors(
                parameters, self.anchor_model_name_or_path
            )
        if parameters.keys() != self._weight_dist_anchors.keys():
            raise RuntimeError(
                "trainable parameters changed after loading weight-distance anchors"
            )
        return parameters, self._weight_dist_anchors


class PretrainedAnchorWeightDistCoefficient01Trainer(PretrainedAnchorWeightDistTrainer):
    """Pretrained-anchor weight distance with coefficient 0.1."""

    distance_coefficient = 0.1


class PretrainedAnchorWeightDistCoefficient10Trainer(PretrainedAnchorWeightDistTrainer):
    """Pretrained-anchor weight distance with coefficient 10."""

    distance_coefficient = 10.0


class PretrainedAnchorWeightDistCoefficient1e5Trainer(
    PretrainedAnchorWeightDistTrainer
):
    """Pretrained-anchor weight distance with coefficient 100,000."""

    distance_coefficient = 1e5


class PretrainedAnchorWeightDistCoefficient1e6Trainer(
    PretrainedAnchorWeightDistTrainer
):
    """Pretrained-anchor weight distance with coefficient 1,000,000."""

    distance_coefficient = 1e6


class WeightDistCoefficient01Trainer(WeightDistTrainer):
    """Weight distance with coefficient 0.1."""

    distance_coefficient = 0.1


class WeightDistCoefficient10Trainer(WeightDistTrainer):
    """Weight distance with coefficient 10."""

    distance_coefficient = 10.0


class WeightDistCoefficient25Trainer(WeightDistTrainer):
    """Weight distance with coefficient 25."""

    distance_coefficient = 25.0


class WeightDistCoefficient50Trainer(WeightDistTrainer):
    """Weight distance with coefficient 50."""

    distance_coefficient = 50.0


class WeightDistCoefficient75Trainer(WeightDistTrainer):
    """Weight distance with coefficient 75."""

    distance_coefficient = 75.0


class WeightDistCoefficient100Trainer(WeightDistTrainer):
    """Weight distance with coefficient 100."""

    distance_coefficient = 100.0
