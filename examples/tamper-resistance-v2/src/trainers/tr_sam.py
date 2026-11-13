"""Sharpness-aware minimization (SAM) applied to unlearning objectives."""

import torch
from torch.func import functional_call

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.ul_gd import GDTrainer
from trainers.utils import BalancedSampler, dict_dot

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class SAMBaseTrainer(AxolotlTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        source_id_forget,
        source_id_retain,
        rho=0.01,
        coeff_retain=1.0,
        coeff_forget=1.0,
        **kwargs,
    ):
        if rho <= 0:
            raise ValueError("rho must be positive")
        if type(source_id_forget) is not int or type(source_id_retain) is not int:
            raise ValueError("source IDs must be integers")
        if source_id_forget == source_id_retain:
            raise ValueError("forget and retain source IDs must differ")
        super().__init__(*args, **kwargs)
        self.source_id_forget = source_id_forget
        self.source_id_retain = source_id_retain
        self.coeff_retain = coeff_retain
        self.coeff_forget = coeff_forget
        self.rho = rho

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the configured forget and retain sources."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 2 or self._train_batch_size % 2:
            raise ValueError("SAM sampling requires an even batch size")

        source_ids = dataset["source_id"]
        sample_ids_forget = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_forget
        ]
        sample_ids_retain = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_retain
        ]
        if not sample_ids_forget or not sample_ids_retain:
            raise ValueError("SAM sampling requires both forget and retain examples")
        return BalancedSampler(sample_ids_forget, sample_ids_retain)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        forget_mask = inputs["source_id"] == self.source_id_forget
        retain_mask = inputs["source_id"] == self.source_id_retain

        x_forget = {field: inputs[field][forget_mask] for field in INP_FIELDS}
        x_retain = {field: inputs[field][retain_mask] for field in INP_FIELDS}

        theta = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Compute a detached perturbation from the forget gradient.
        loss_forget = self.compute_loss_forget(model, x_forget)
        grad_forget = torch.autograd.grad(loss_forget, tuple(theta.values()))
        grad_dict = dict(zip(theta, grad_forget))
        grad_norm = dict_dot(grad_dict, grad_dict).sqrt() + 1e-6
        theta_perturbed = {
            name: param + (self.rho * grad / grad_norm.to(param.device)).detach()
            for (name, param), grad in zip(theta.items(), grad_forget)
        }

        # Evaluate at perturbed weights while keeping gradients connected to theta.
        loss_forget = self.compute_loss_forget(
            lambda **x: functional_call(model, theta_perturbed, (), x), x_forget
        )

        # The model's original weights remain unchanged.
        loss_retain = self.compute_loss_retain(model, x_retain)

        self.log(
            {
                "loss_retain": loss_retain.detach().item(),
                "loss_forget": loss_forget.detach().item(),
            }
        )

        loss = self.coeff_retain * loss_retain + self.coeff_forget * loss_forget
        return (loss, None) if return_outputs else loss


class SAMGDTrainer(SAMBaseTrainer, GDTrainer):
    pass
