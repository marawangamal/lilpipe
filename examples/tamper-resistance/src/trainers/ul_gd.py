"""Plain gradient-difference unlearning with balanced forget and retain batches."""

import torch

from trainers._base_unlearning import UnlearningTrainer

from trainers.utils import (
    AuthorsMixedSourceSampler,
    MixedSourceSampler,
    OptmlMixedSourceSampler,
    global_l2_norm,
)


class GradDiffTrainer(UnlearningTrainer):
    """Minimize retain cross-entropy while maximizing forget cross-entropy."""

    def __init__(
        self,
        *args,
        sampling="balanced",
        **kwargs,
    ):
        if sampling not in {"balanced", "authors", "optml"}:
            raise ValueError("sampling must be 'balanced', 'authors', or 'optml'")
        super().__init__(*args, **kwargs)
        self.sampling = sampling

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self.sampling == "optml":
            return OptmlMixedSourceSampler(
                dataset,
                self.args.per_device_train_batch_size,
                interleave=True,
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
            interleave=True,
        )

    def compute_forget_loss(
        self,
        model,
        inputs,
        return_grads=False,
        return_outputs=False,
        model_kwargs=None,
    ):
        return self.compute_loss_term(
            model,
            inputs,
            -1.0,
            return_grads,
            return_outputs,
            model_kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {name: value for name, value in inputs.items() if name != "is_forget"}
        inp_forget = {name: value[0::2] for name, value in inputs.items()}
        inp_retain = {name: value[1::2] for name, value in inputs.items()}
        loss_forget, forget_outputs = self.compute_forget_loss(
            model, inp_forget, return_outputs=True
        )
        loss_forget = self.coeff_forget * loss_forget
        loss_retain = self.coeff_retain * self.compute_retain_loss(model, inp_retain)
        loss = loss_forget + loss_retain

        if model.training and torch.is_grad_enabled():
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
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
            self.log(
                {
                    "loss_forget": (-loss_forget).detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "grad_norm_forget": global_l2_norm(
                        gradient for gradient in grad_forget if gradient is not None
                    ).item(),
                    "grad_norm_retain": global_l2_norm(
                        gradient for gradient in grad_retain if gradient is not None
                    ).item(),
                }
            )
        return (loss, forget_outputs) if return_outputs else loss
