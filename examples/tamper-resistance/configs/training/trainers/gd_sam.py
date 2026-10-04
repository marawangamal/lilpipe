"""Gradient-difference unlearning with SAM on the forget objective."""

import torch

from configs.training.trainers.mamul import MAMULTrainer
from configs.training.trainers.samplers import MixedSourceSampler


class SAMAttack(torch.nn.Module):
    def __init__(self, rho=0.01):
        super().__init__()
        self.rho = rho

    def forward(self, parameters, loss):
        gradients = torch.autograd.grad(
            loss, tuple(parameters.values()), allow_unused=True
        )
        active_gradients = [gradient for gradient in gradients if gradient is not None]
        if not active_gradients:
            return {}
        norm = torch.stack(
            [gradient.detach().float().norm() for gradient in active_gradients]
        ).norm()
        scale = self.rho / norm.clamp_min(1e-12)
        return {
            name: (scale * gradient).detach()
            for (name, _), gradient in zip(parameters.items(), gradients)
            if gradient is not None
        }


class GDSAMTrainer(MAMULTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        rho=0.01,
        retain_coeff=1.0,
        forget_coeff=1.0,
        attack_type="sam",
        attack_kwargs=None,
        **kwargs,
    ):
        if attack_type.lower() != "sam":
            raise ValueError("GDSAMTrainer only supports attack_type='sam'")
        super().__init__(
            *args, retain_coeff=retain_coeff, forget_coeff=forget_coeff, **kwargs
        )
        self.attack = SAMAttack(**{"rho": rho, **(attack_kwargs or {})})

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

    def compute_forget_loss(self, model_forward, inputs, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        outputs = model_forward(**forget_inputs)
        return -outputs.loss.float()

    def compute_retain_loss(self, model_forward, inputs, **kwargs):
        _, retain_inputs = self.split_sources(inputs)
        outputs = model_forward(**retain_inputs)
        return outputs.loss.float()

    def sample_attack(self, model, parameters, inputs, **kwargs):
        loss = self.compute_forget_loss(model, inputs, **kwargs)
        return self.attack(parameters, loss)


# Compatibility alias for existing config class paths.
GradDiffTrainer = GDSAMTrainer
