"""SAM-based unlearning with balanced forget and retain batches."""

import torch
from axolotl.core.trainers.base import AxolotlTrainer
from torch.func import functional_call

from configs.training.trainers.samplers import MixedSourceSampler


class SAMAttack(torch.nn.Module):
    def __init__(self, rho=0.01):
        super().__init__()
        self.rho = rho

    def forward(self, parameters, loss):
        gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        norm = torch.stack(
            [gradient.detach().float().norm() for gradient in gradients]
        ).norm()
        scale = self.rho / norm.clamp_min(1e-12)

        return {
            # We used a stopgrad, so we have
            # θ_attack = θ + const
            # therefore,
            # dθ_attack/dθ = I + 0 (i.e., pass-through-estimator)
            name: parameter + scale * gradient.detach()
            for (name, parameter), gradient in zip(parameters.items(), gradients)
        }


class TARTrainer(AxolotlTrainer):
    def __init__(
        self,
        *args,
        rho=0.01,
        retain_coeff=1.0,
        forget_coeff=1.0,
        attack_type="none",
        attack_kwargs=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sam_kwargs = {"rho": rho, **(attack_kwargs or {})}
        self.attack = {
            "sam": SAMAttack(**sam_kwargs),
            "none": lambda parameters, loss: parameters,
            # "tar" : ...
        }[attack_type.lower()]
        self.retain_coeff = retain_coeff
        self.forget_coeff = forget_coeff

    def compute_retain_loss(self, model_forward, inputs, **kwargs):
        raise NotImplementedError

    def compute_forget_loss(self, model_forward, inputs, **kwargs):
        raise NotImplementedError

    def compute_forget_loss_at_params(self, model, inputs, parameters, **kwargs):
        def model_forward(**model_inputs):
            return functional_call(model, parameters, (), model_inputs)

        return self.compute_forget_loss(model_forward, inputs, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        loss_forget = self.compute_forget_loss(model, inputs, **kwargs)
        theta_prime = self.attack(parameters, loss_forget)

        loss_forget = self.compute_forget_loss_at_params(
            model, inputs, theta_prime, **kwargs
        )
        loss_retain = self.compute_retain_loss(model, inputs, **kwargs)
        loss = self.forget_coeff * loss_forget + self.retain_coeff * loss_retain
        return (loss, None) if return_outputs else loss


class GradDiffTrainer(TARTrainer):
    """Apply SAM to gradient-difference unlearning's forget objective."""

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
