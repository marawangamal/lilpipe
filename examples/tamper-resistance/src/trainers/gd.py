"""Plain gradient-difference unlearning with balanced forget and retain batches."""

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.gradient_metrics import global_l2_norm
from trainers.sampling import (
    AuthorsMixedSourceSampler,
    MixedSourceSampler,
    OptmlMixedSourceSampler,
)


class GradDiffTrainer(AxolotlTrainer):
    """Minimize retain cross-entropy while maximizing forget cross-entropy."""

    def __init__(
        self,
        *args,
        forget_coefficient=1.0,
        retain_coefficient=1.0,
        sampling="balanced",
        **kwargs,
    ):
        if sampling not in {"balanced", "authors", "optml"}:
            raise ValueError("sampling must be 'balanced', 'authors', or 'optml'")
        super().__init__(*args, **kwargs)
        self.forget_coefficient = forget_coefficient
        self.retain_coefficient = retain_coefficient
        self.sampling = sampling

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("GradDiff requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}

        forget_outputs = model(**forget_inputs)
        forget_ce = forget_outputs.loss.float()
        retain_ce = model(**retain_inputs).loss.float()
        forget_term = -self.forget_coefficient * forget_ce
        retain_term = self.retain_coefficient * retain_ce
        loss = forget_term + retain_term

        if model.training and torch.is_grad_enabled():
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            forget_gradients = torch.autograd.grad(
                forget_term,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            retain_gradients = torch.autograd.grad(
                retain_term,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            self.log(
                {
                    "forget_ce": forget_ce.detach().item(),
                    "retain_ce": retain_ce.detach().item(),
                    "forget_term": forget_term.detach().item(),
                    "retain_term": retain_term.detach().item(),
                    "forget_grad_norm": global_l2_norm(
                        gradient
                        for gradient in forget_gradients
                        if gradient is not None
                    ).item(),
                    "retain_grad_norm": global_l2_norm(
                        gradient
                        for gradient in retain_gradients
                        if gradient is not None
                    ).item(),
                }
            )
        return (loss, forget_outputs) if return_outputs else loss
