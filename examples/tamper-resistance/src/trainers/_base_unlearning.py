"""Shared unlearning coefficients and unweighted retain loss."""

import math

import torch
from axolotl.core.trainers.base import AxolotlTrainer


class UnlearningTrainer(AxolotlTrainer):
    def __init__(self, *args, coeff_forget=1.0, coeff_retain=1.0, **kwargs):
        if any(not math.isfinite(c) or c < 0 for c in (coeff_forget, coeff_retain)):
            raise ValueError("Unlearning coefficients must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self.coeff_forget = coeff_forget
        self.coeff_retain = coeff_retain

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

    def compute_retain_loss(
        self, model, inputs, return_grads=False, return_outputs=False
    ):
        """Return retain CE without applying its objective weight."""
        return self.compute_loss_term(model, inputs, 1.0, return_grads, return_outputs)
