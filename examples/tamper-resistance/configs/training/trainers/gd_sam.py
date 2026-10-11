"""Authors-matched gradient-difference unlearning with SAM."""

import random
import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()

from axolotl.core.trainers.base import AxolotlTrainer

from configs.training.trainers.samplers import MixedSourceSampler


class AuthorsMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Shuffle forget rows and draw retain rows with replacement."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("mixed source sampling requires an even batch size")
        sources = dataset["is_forget"]
        self.forget = [index for index, value in enumerate(sources) if value]
        self.retain = [index for index, value in enumerate(sources) if not value]
        if not self.forget or not self.retain:
            raise ValueError("mixed source sampling requires forget and retain rows")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        forget = self.forget.copy()
        rng.shuffle(forget)
        half_batch = self.batch_size // 2
        usable = len(forget) // half_batch * half_batch
        for start in range(0, usable, half_batch):
            batch = forget[start : start + half_batch]
            batch += [rng.choice(self.retain) for _ in range(half_batch)]
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        half_batch = self.batch_size // 2
        return 2 * (len(self.forget) // half_batch * half_batch)


class AuthorsSamplingMixin:
    """Use the paper code's sampling policy."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return AuthorsMixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.seed,
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


class GDSAMTrainer(AxolotlTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    rho = 0.01

    def __init__(
        self,
        *args,
        retain_coeff=1.0,
        forget_coeff=1.0,
        attack_type="sam",
        attack_kwargs=None,
        **kwargs,
    ):
        if attack_type.lower() != "sam":
            raise ValueError("GDSAMTrainer only supports attack_type='sam'")
        attack_kwargs = attack_kwargs or {}
        unknown_attack_kwargs = set(attack_kwargs) - {"rho"}
        if unknown_attack_kwargs:
            raise ValueError(
                f"unsupported SAM attack arguments: {sorted(unknown_attack_kwargs)}"
            )
        super().__init__(*args, **kwargs)
        self.retain_coeff = retain_coeff
        self.forget_coeff = forget_coeff
        self.rho = attack_kwargs.get("rho", self.rho)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
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
            perturbation = sam_perturbation(forget_gradients, self.rho)

            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                forget_loss, perturbed_forget_gradients = (
                    self._backward_and_take_gradients(
                        lambda: self.forget_coeff
                        * self.compute_forget_loss(model, inputs),
                        parameters,
                    )
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            retain_loss, retain_gradients = self._backward_and_take_gradients(
                lambda: self.retain_coeff * self.compute_retain_loss(model, inputs),
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


class GDSAMRho1eMinus3Trainer(AuthorsSamplingMixin, GDSAMTrainer):
    """Authors-matched GD+SAM with rho fixed to 1e-3."""

    rho = 1e-3


class GDSAMRho2p5eMinus3Trainer(GDSAMTrainer):
    rho = 2.5e-3


class GDSAMRho5eMinus3Trainer(GDSAMTrainer):
    rho = 5e-3


class GDSAMPlainRho1eMinus2Trainer(GDSAMTrainer):
    rho = 1e-2


class GDSAMAuthorsRho5eMinus3Trainer(AuthorsSamplingMixin, GDSAMTrainer):
    rho = 5e-3


class GDSAMRho1eMinus2Trainer(AuthorsSamplingMixin, GDSAMTrainer):
    """Authors-matched GD+SAM with rho fixed to 1e-2."""

    rho = 1e-2


class GDSAMRho1eMinus1Trainer(AuthorsSamplingMixin, GDSAMTrainer):
    """Authors-matched GD+SAM with rho fixed to 1e-1."""

    rho = 1e-1


# Compatibility alias for existing config class paths.
GradDiffTrainer = GDSAMTrainer
