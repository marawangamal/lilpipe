"""Representation Misdirection for Unlearning."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

from trainers.activation_orthogonality import ActivationOrthogonalityMixin
from trainers.gd_sam import sam_perturbation
from trainers.gradient_metrics import gradient_dict_l2_norm
from trainers.sampling import MixedSourceSampler
from trainers.weight_distance import global_rms_distance, load_reference_parameters


class RMUTrainer(AxolotlTrainer):
    """Steer forget activations while preserving retain activations."""

    def __init__(
        self,
        *args,
        reference_model_name_or_path,
        retain_coefficient=1200.0,
        steering_coefficient=6.5,
        target_layer=7,
        control_seed=42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if retain_coefficient < 0:
            raise ValueError("retain_coefficient must be non-negative")
        if steering_coefficient <= 0:
            raise ValueError("steering_coefficient must be positive")
        if target_layer < 0 or target_layer >= self.model.config.num_hidden_layers:
            raise ValueError("target_layer must identify a model layer")
        self.reference_model_name_or_path = reference_model_name_or_path
        self.retain_coefficient = retain_coefficient
        self.steering_coefficient = steering_coefficient
        self.target_layer = target_layer
        self.control_seed = control_seed
        self._theta_ref = None
        self._control_vector = None
        self._rmu_logged_step = None

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

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
                "trainable parameters changed after loading the reference"
            )
        return self._theta_ref

    @contextmanager
    def reference_parameters(self):
        theta = self.theta
        theta_current = {name: parameter.data for name, parameter in theta.items()}
        try:
            for name, parameter in theta.items():
                parameter.data = self.theta_ref[name]
            yield
        finally:
            for name, parameter in theta.items():
                parameter.data = theta_current[name]

    @staticmethod
    def split_sources(inputs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("RMU requires forget and retain rows in every batch")
        fields = ("input_ids", "attention_mask")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return forget_inputs, retain_inputs

    def activation(self, model, inputs):
        hidden_states = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        return hidden_states[self.target_layer + 1]

    def reference_activation(self, model, inputs):
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad(), self.reference_parameters():
                return self.activation(model, inputs).detach()
        finally:
            model.train(was_training)

    def control_vector(self, activation):
        if self._control_vector is None:
            generator = torch.Generator(device=activation.device)
            generator.manual_seed(self.control_seed)
            vector = torch.rand(
                1,
                1,
                activation.shape[-1],
                device=activation.device,
                dtype=activation.dtype,
                generator=generator,
            )
            self._control_vector = (
                F.normalize(vector.float(), dim=-1) * self.steering_coefficient
            ).to(activation.dtype)
        return self._control_vector

    @staticmethod
    def activation_loss(activation, target, attention_mask):
        squared_difference = F.mse_loss(
            activation.float(), target.float(), reduction="none"
        ).mean(dim=-1)
        mask = attention_mask.to(
            device=squared_difference.device, dtype=squared_difference.dtype
        )
        return (
            (squared_difference * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        ).mean()

    def compute_forget_loss(self, model, inputs):
        forget_inputs, _ = self.split_sources(inputs)
        hidden_states = model(
            **forget_inputs,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        return self.compute_forget_loss_from_hidden_states(hidden_states, forget_inputs)

    def compute_forget_loss_from_hidden_states(self, hidden_states, forget_inputs):
        activation = hidden_states[self.target_layer + 1]
        return self.activation_loss(
            activation,
            self.control_vector(activation),
            forget_inputs["attention_mask"],
        )

    def compute_retain_loss(self, model, inputs):
        _, retain_inputs = self.split_sources(inputs)
        activation_ref = self.reference_activation(model, retain_inputs)
        activation = self.activation(model, retain_inputs)
        return self.retain_coefficient * self.activation_loss(
            activation,
            activation_ref,
            retain_inputs["attention_mask"],
        )

    def compute_rmu_loss(self, model, inputs, log_metrics=True):
        loss_forget = self.compute_forget_loss(model, inputs)
        loss_retain = self.compute_retain_loss(model, inputs)
        loss = loss_forget + loss_retain

        step = self.state.global_step
        if log_metrics and step != self._rmu_logged_step:
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                }
            )
            self._rmu_logged_step = step
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = self.compute_rmu_loss(model, inputs)
        return (loss, {}) if return_outputs else loss


class RMUAOTrainer(ActivationOrthogonalityMixin, RMUTrainer):
    """Add forget-set activation orthogonality to RMU."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        if forget_inputs["input_ids"].shape[0] < 2:
            raise ValueError("RMU+AO requires at least two forget rows per batch")

        hidden_states = model(
            **forget_inputs,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        loss_forget = self.compute_forget_loss_from_hidden_states(
            hidden_states, forget_inputs
        )
        loss_retain = self.compute_retain_loss(model, inputs)
        loss_activation_orthogonality = self.compute_activation_orthogonality_loss(
            hidden_states, forget_inputs["attention_mask"]
        )
        loss = loss_forget + loss_retain + loss_activation_orthogonality

        step = self.state.global_step
        if step != self._rmu_logged_step:
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "loss_activation_orthogonality": (
                        loss_activation_orthogonality.detach().item()
                    ),
                }
            )
            self._rmu_logged_step = step

        outputs = {
            "loss_forget": loss_forget,
            "loss_retain": loss_retain,
            "loss_activation_orthogonality": loss_activation_orthogonality,
        }
        return (loss, outputs) if return_outputs else loss


class RMUSAMTrainer(RMUTrainer):
    """Apply SAM to RMU's forget objective."""

    def __init__(self, *args, rho=0.01, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        super().__init__(*args, **kwargs)
        self.rho = rho

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
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

        parameters = self.theta
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, gradients_at_perturbation = self._backward_and_take_gradients(
                lambda: self.compute_forget_loss(model, inputs), parameters
            )
            perturbation = sam_perturbation(gradients_at_perturbation, self.rho)

            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                loss_forget, gradients_forget = self._backward_and_take_gradients(
                    lambda: self.compute_forget_loss(model, inputs), parameters
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            loss_retain, gradients_retain = self._backward_and_take_gradients(
                lambda: self.compute_retain_loss(model, inputs), parameters
            )
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "grad_norm_forget": gradient_dict_l2_norm(gradients_forget).item(),
                    "grad_norm_retain": gradient_dict_l2_norm(gradients_retain).item(),
                    "grad_norm_forget_at_perturbation": gradient_dict_l2_norm(
                        gradients_at_perturbation
                    ).item(),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            for gradient in (gradients_forget[name], gradients_retain[name]):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )

        return (
            (loss_forget + loss_retain) / self.current_gradient_accumulation_steps
        ).detach()


class RMUWeightDistanceTrainer(RMUTrainer):
    """Apply RMU while maximizing RMS distance from the reference model."""

    def __init__(self, *args, coefficient_weight_dist=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.coefficient_weight_dist = coefficient_weight_dist
        self._weight_distance_logged_step = None

    def compute_weight_dist_loss(self):
        return -self.coefficient_weight_dist * global_rms_distance(
            self.theta.values(), self.theta_ref.values()
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_rmu = self.compute_rmu_loss(model, inputs)
        loss_weight_dist = self.compute_weight_dist_loss()
        loss = loss_rmu + loss_weight_dist

        step = self.state.global_step
        if step != self._weight_distance_logged_step:
            self.log({"loss_weight_dist": loss_weight_dist.detach().item()})
            self._weight_distance_logged_step = step
        return (loss, {}) if return_outputs else loss
