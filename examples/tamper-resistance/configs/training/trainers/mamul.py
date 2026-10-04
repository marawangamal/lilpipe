"""Composable meta-adversarial model-unlearning trainer."""

from collections.abc import Mapping

import torch
from axolotl.core.trainers.base import AxolotlTrainer
from torch.func import functional_call


class MAMULTrainer(AxolotlTrainer):
    def __init__(self, *args, retain_coeff=1.0, forget_coeff=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_coeff = retain_coeff
        self.forget_coeff = forget_coeff

    def compute_retain_loss(self, model_forward, inputs, **kwargs):
        raise NotImplementedError

    def compute_forget_loss(self, model_forward, inputs, **kwargs):
        raise NotImplementedError

    def sample_attack(self, model, parameters, inputs, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _validate_delta(parameters, delta):
        if not isinstance(delta, Mapping):
            raise TypeError("sample_attack must return a parameter-delta mapping")
        unknown = set(delta) - set(parameters)
        if unknown:
            raise ValueError(
                f"attack delta contains unknown parameters: {sorted(unknown)}"
            )
        for name, change in delta.items():
            parameter = parameters[name]
            if not isinstance(change, torch.Tensor):
                raise TypeError(f"attack delta for {name!r} is not a tensor")
            if change.shape != parameter.shape:
                raise ValueError(
                    f"attack delta for {name!r} has shape {tuple(change.shape)}, "
                    f"expected {tuple(parameter.shape)}"
                )
            if change.device != parameter.device:
                raise ValueError(
                    f"attack delta for {name!r} is on {change.device}, "
                    f"expected {parameter.device}"
                )
            if change.dtype != parameter.dtype:
                raise ValueError(
                    f"attack delta for {name!r} has dtype {change.dtype}, "
                    f"expected {parameter.dtype}"
                )
            if change.requires_grad:
                raise ValueError(f"attack delta for {name!r} must be detached")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        delta = self.sample_attack(model, parameters, inputs, **kwargs)
        self._validate_delta(parameters, delta)
        attacked = {
            name: parameter + delta.get(name, torch.zeros_like(parameter))
            for name, parameter in parameters.items()
        }

        def attacked_forward(**model_inputs):
            return functional_call(model, attacked, (), model_inputs)

        loss_forget = self.compute_forget_loss(attacked_forward, inputs, **kwargs)
        loss_retain = self.compute_retain_loss(model, inputs, **kwargs)
        loss = self.forget_coeff * loss_forget + self.retain_coeff * loss_retain
        return (loss, None) if return_outputs else loss
