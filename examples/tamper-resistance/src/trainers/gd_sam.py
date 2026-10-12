"""Authors-matched gradient-difference unlearning with SAM."""

import torch
import torch.distributed as dist
from transformers import enable_full_determinism

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.sampling import (
    AuthorsMixedSourceSampler,
    MixedSourceSampler,
    OptmlMixedSourceSampler,
)


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
            raise RuntimeError(
                "DTensor SAM gradients require distributed initialization"
            )
        dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    return local_squared_norm.sqrt()


def sam_perturbation(gradients, rho):
    """Return a radius-rho step from synchronized parameter gradients."""
    active_gradients = [
        gradient for gradient in gradients.values() if gradient is not None
    ]
    if not active_gradients:
        raise RuntimeError("SAM forget backward produced no parameter gradients")
    norm = global_l2_norm(active_gradients)
    if not norm.isfinite() or norm == 0:
        raise RuntimeError(f"SAM forget gradient has invalid norm: {norm}")
    return {
        name: (rho * gradient / norm.to(gradient.device)).detach()
        for name, gradient in gradients.items()
        if gradient is not None
    }


def optml_sam_perturbation(gradients, rho):
    """Reproduce OPTML's original-dtype, single-device norm calculation."""
    active_gradients = [
        gradient for gradient in gradients.values() if gradient is not None
    ]
    if not active_gradients:
        raise RuntimeError("SAM forget backward produced no parameter gradients")

    norm_device = active_gradients[0].device
    per_parameter_norms = [
        gradient.to(norm_device).norm(2) for gradient in active_gradients
    ]
    norm = torch.stack(per_parameter_norms).norm(2)
    if not norm.isfinite() or norm == 0:
        raise RuntimeError(f"SAM forget gradient has invalid norm: {norm}")
    return {
        name: (rho * gradient / norm.to(gradient.device)).detach()
        for name, gradient in gradients.items()
        if gradient is not None
    }


class GDSAMTrainer(AxolotlTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        rho=0.01,
        retain_coefficient=1.0,
        forget_coefficient=1.0,
        sampling="balanced",
        optml_parity=False,
        **kwargs,
    ):
        if rho <= 0:
            raise ValueError("rho must be positive")
        if sampling not in {"balanced", "authors", "optml"}:
            raise ValueError("sampling must be 'balanced', 'authors', or 'optml'")
        super().__init__(*args, **kwargs)
        self.retain_coefficient = retain_coefficient
        self.forget_coefficient = forget_coefficient
        self.rho = rho
        self.sampling = sampling
        self.optml_parity = optml_parity
        if self.optml_parity:
            enable_full_determinism(self.args.seed)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self.sampling == "optml":
            return OptmlMixedSourceSampler(
                dataset,
                self.args.per_device_train_batch_size,
            )
        sampler_cls = (
            AuthorsMixedSourceSampler
            if self.sampling == "authors"
            else MixedSourceSampler
        )
        return sampler_cls(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @staticmethod
    def split_sources(inputs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("GradDiff requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return forget_inputs, retain_inputs

    def compute_forget_loss(self, model, inputs, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        outputs = model(**forget_inputs)
        return -outputs.loss.float()

    def compute_retain_loss(self, model, inputs, **kwargs):
        _, retain_inputs = self.split_sources(inputs)
        outputs = model(**retain_inputs)
        return outputs.loss.float()

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
        """Run FSDP-aware backward and remove its gradients from the model."""
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, forget_gradients = self._backward_and_take_gradients(
                lambda: self.compute_forget_loss(model, inputs), parameters
            )
            perturbation_fn = (
                optml_sam_perturbation if self.optml_parity else sam_perturbation
            )
            perturbation = perturbation_fn(forget_gradients, self.rho)

            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                forget_loss, perturbed_forget_gradients = (
                    self._backward_and_take_gradients(
                        lambda: self.forget_coefficient
                        * self.compute_forget_loss(model, inputs),
                        parameters,
                    )
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            retain_loss, retain_gradients = self._backward_and_take_gradients(
                lambda: self.retain_coefficient
                * self.compute_retain_loss(model, inputs),
                parameters,
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            for gradient in (
                perturbed_forget_gradients[name],
                retain_gradients[name],
            ):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )

        return (
            (forget_loss + retain_loss) / self.current_gradient_accumulation_steps
        ).detach()
