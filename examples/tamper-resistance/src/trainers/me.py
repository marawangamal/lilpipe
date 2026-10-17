"""Maximum-entropy unlearning with balanced forget and retain batches."""

import math

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.gd_sam import sam_perturbation
from trainers.gradient_metrics import global_l2_norm, gradient_dict_l2_norm
from trainers.sampling import (
    AuthorsMixedSourceSampler,
    MixedSourceSampler,
    OptmlMixedSourceSampler,
)


class MaxEntTrainer(AxolotlTrainer):
    """Minimize retain cross-entropy while maximizing forget entropy."""

    def __init__(
        self,
        *args,
        forget_coefficient=1.0,
        retain_coefficient=1.0,
        sampling="balanced",
        log_gradient_norms=False,
        **kwargs,
    ):
        if sampling not in {"balanced", "authors", "optml"}:
            raise ValueError("sampling must be 'balanced', 'authors', or 'optml'")
        super().__init__(*args, **kwargs)
        self.forget_coefficient = forget_coefficient
        self.retain_coefficient = retain_coefficient
        self.sampling = sampling
        self.log_gradient_norms = log_gradient_norms

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

    def compute_loss_term(
        self,
        model,
        inputs,
        coefficient,
        return_grads=False,
        return_outputs=False,
        model_kwargs=None,
    ):
        outputs = model(**inputs, **(model_kwargs or {}))
        loss = coefficient * outputs.loss.float()

        if return_grads:
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if return_outputs:
                return loss, grads, outputs
            return loss, grads
        if return_outputs:
            return loss, outputs
        return loss

    def compute_forget_loss(
        self,
        model,
        inputs,
        return_grads=False,
        return_outputs=False,
        model_kwargs=None,
    ):
        forget_mask = inputs["is_forget"].bool()
        if not forget_mask.any():
            raise ValueError("MaxEnt requires forget and retain rows in every batch")
        forget_inputs = {
            field: inputs[field][forget_mask]
            for field in ("input_ids", "attention_mask")
        }
        forget_labels = inputs["labels"][forget_mask]
        outputs = model(**forget_inputs, **(model_kwargs or {}))
        log_probs = torch.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
        probs = torch.exp(log_probs)
        negative_entropy_per_token = torch.sum(probs * log_probs, dim=-1)
        valid = forget_labels[:, 1:] != -100
        if not valid.any():
            raise ValueError("MaxEnt requires a supervised token in every forget batch")
        entropy = -torch.mean(negative_entropy_per_token[valid])
        loss = -self.forget_coefficient * entropy
        self._last_forget_entropy = entropy.detach()

        if return_grads:
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if return_outputs:
                return loss, grads, outputs
            return loss, grads
        if return_outputs:
            return loss, outputs
        return loss

    def compute_retain_loss(
        self, model, inputs, return_grads=False, return_outputs=False
    ):
        retain_mask = ~inputs["is_forget"].bool()
        if not retain_mask.any():
            raise ValueError("MaxEnt requires forget and retain rows in every batch")
        fields = ("input_ids", "attention_mask", "labels")
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return self.compute_loss_term(
            model,
            retain_inputs,
            self.retain_coefficient,
            return_grads,
            return_outputs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_forget, forget_outputs = self.compute_forget_loss(
            model, inputs, return_outputs=True
        )
        loss_retain = self.compute_retain_loss(model, inputs)
        loss = loss_forget + loss_retain

        if model.training and torch.is_grad_enabled():
            metrics = {
                "entropy_forget": self._last_forget_entropy.item(),
                "entropy_ceiling": math.log(forget_outputs.logits.shape[-1]),
                "loss_retain": loss_retain.detach().item(),
            }
            if self.log_gradient_norms:
                parameters = tuple(
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                grad_forget = torch.autograd.grad(
                    loss_forget,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_retain = torch.autograd.grad(
                    loss_retain,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                metrics.update(
                    {
                        "grad_norm_forget": global_l2_norm(
                            gradient for gradient in grad_forget if gradient is not None
                        ).item(),
                        "grad_norm_retain": global_l2_norm(
                            gradient for gradient in grad_retain if gradient is not None
                        ).item(),
                    }
                )
            self.log(metrics)
        return (loss, forget_outputs) if return_outputs else loss


class MaxEntSAMTrainer(MaxEntTrainer):
    """Apply SAM to MaxEnt's forget objective."""

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
                    "entropy_forget": self._last_forget_entropy.item(),
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
