"""Gradient-difference loss terms for unlearning."""


import torch
from torch.func import functional_call

from axolotl.core.trainers.base import AxolotlTrainer
from trainers.utils import dict_dot, dict_binop, model_to_params, BalancedSampler
from trainers.ul_gd import GDTrainer

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class GMABaseTrackTrainer(AxolotlTrainer):
    def __init__(
        self,
        *args,
        source_id_cosim_a,
        source_id_cosim_b,
        eps_finite_diff=1e-3,
        eps_norm_div=1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.source_id_cosim_a = source_id_cosim_a
        self.source_id_cosim_b = source_id_cosim_b
        self.eps_norm_div = eps_norm_div
        self.eps_finite_diff = eps_finite_diff

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the configured forget and retain sources."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 2 or self._train_batch_size % 2:
            raise ValueError("SAM sampling requires an even batch size")

        source_ids = dataset["source_id"]
        source_id_cosim_a = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_cosim_a
        ]
        source_id_cosim_b = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_cosim_b
        ]
        if not source_id_cosim_a or not source_id_cosim_b:
            raise ValueError("Cosim sampling requires both forget and retain examples")
        return BalancedSampler(source_id_cosim_a, source_id_cosim_b)

    def compute_loss_avg_of_cosims(self, model, forget, compute_loss_only=False):
        """Average cosine over all distinct pairs of per-example CE gradients.
        
        NOTE: this approximation may not work properly if dropout is enabled.
        """
        B = len(forget["labels"])
        if B < 2:
            raise ValueError("GIP requires at least two forget examples")
        theta = model_to_params(model, only_trainable=True)

        # compute m (i.e., the mean of normalized gradients)
        m = {name: torch.zeros_like(p) for name, p in theta.items()}
        self_cosines = 0
        for i in range(B):
            batch = {name: value[i : i + 1] for name, value in forget.items()}
            g_theta_i = dict(
                zip(
                    theta,
                    torch.autograd.grad(
                        model(**batch).loss
                        , tuple(theta.values())
                    ),
                )
            )
            norm_i = dict_dot(g_theta_i, g_theta_i).sqrt()
            denom_i = norm_i + self.eps_norm_div
            self_cosines += (norm_i / denom_i).square()
            for name, p in theta.items():
                m[name] += g_theta_i[name] / (B * denom_i.to(p.device))

        cosine = (B**2 * dict_dot(m, m) - self_cosines) / (B * (B - 1))
        if compute_loss_only:
            return cosine

        # estimate gradient from minibatch
        g_hat = None
        for i in range(B):
            batch = {name: value[i : i + 1] for name, value in forget.items()}

            # compute v_i
            g_theta_i = dict(
                zip(
                    theta,
                    torch.autograd.grad(
                        model(**batch).loss, tuple(theta.values())
                    ),
                )
            )
            norm_i = dict_dot(g_theta_i, g_theta_i).sqrt()
            denom_i = norm_i + self.eps_norm_div
            u_i = {
                name: value / denom_i.to(value.device)
                for name, value in g_theta_i.items()
            }
            m_others = dict_binop(m, u_i, lambda mean, u: mean - u / B)
            norm_i_safe = norm_i.clamp_min(torch.finfo(norm_i.dtype).tiny)
            projection = dict_dot(m_others, u_i) / norm_i_safe
            v_i = {
                name: m_others[name] / denom_i.to(p.device)
                - projection.to(p.device) * u_i[name]
                for name, p in theta.items()
            }

            theta_shifted = {
                name: (
                    p.detach() + self.eps_finite_diff * v_i[name].detach()
                ).requires_grad_(True)
                for name, p in theta.items()
            }

            loss_shifted = functional_call(
                model, theta_shifted, (), batch
            ).loss

            g_theta_i_plus_eps_v = dict(zip(
                theta_shifted,
                torch.autograd.grad(
                    loss_shifted, tuple(theta_shifted.values())
                ),
            ))

            difference = dict_binop(
                g_theta_i,
                g_theta_i_plus_eps_v,
                lambda original, shifted: (shifted - original) / B,
            )
            if g_hat is None:
                g_hat = difference
            else: # accumulate grad
                g_hat = dict_binop(g_hat, difference, lambda a, b: a + b)

        # Use surrogate loss. We want to accumulate g_hat. The basic idea is if we
        # return d/dtheta (const_cosim + const_g * theta) = const_g 

        scale = 2 * B / ((B - 1) * self.eps_finite_diff)
        surrogate_loss = sum(
            (theta[name] * (scale * g_hat[name]).detach()).sum()
            for name in theta
        )

        return cosine.detach() + (surrogate_loss - surrogate_loss.detach())

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch = {field: inputs[field] for field in INP_FIELDS}
        if model.training and len(batch["labels"]) < 2:
            raise ValueError("GMABaseTrackTrainer requires at least two samples")
        
        result = super().compute_loss(
            model, batch, return_outputs=return_outputs, **kwargs
        )
        cosine = self.compute_loss_avg_of_cosims(
            model, batch, compute_loss_only=True
        )
        self.log({"grad_cosine": cosine.item()})

        return result


class GMABaseTrainer(GMABaseTrackTrainer):
    def __init__(
        self,
        *args,
        source_id_forget,
        source_id_retain,
        coeff_retain=1.0,
        coeff_forget=1.0,
        coeff_gma=1.0,
        eps_finite_diff=1e-3,
        eps_norm_div=1e-8,
        **kwargs,
    ):
        super().__init__(
            *args,
            source_id_cosim_a=source_id_forget,
            source_id_cosim_b=source_id_retain,
            eps_finite_diff=eps_finite_diff,
            eps_norm_div=eps_norm_div,
            **kwargs,
        )
        self.source_id_forget = source_id_forget
        self.source_id_retain = source_id_retain
        self.coeff_retain = coeff_retain
        self.coeff_forget = coeff_forget
        self.coeff_gma = coeff_gma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        forget_mask = inputs["source_id"] == self.source_id_forget
        retain_mask = inputs["source_id"] == self.source_id_retain

        x_forget = {field: inputs[field][forget_mask] for field in INP_FIELDS}
        x_retain = {field: inputs[field][retain_mask] for field in INP_FIELDS}

        # Compute losses
        loss_forget = self.compute_loss_forget(model, x_forget)
        loss_retain = self.compute_loss_retain(model, x_retain)
        avg_of_cosims = self.compute_loss_avg_of_cosims(model, x_forget)

        self.log(
            {
                "loss_retain": loss_retain.detach().item(),
                "loss_forget": loss_forget.detach().item(),
                "avg_of_cosims": avg_of_cosims.detach().item()
            }
        )

        loss = self.coeff_retain * loss_retain + self.coeff_forget * loss_forget + self.coeff_gma * avg_of_cosims
        return (loss, None) if return_outputs else loss



class GMAGDTrainer(GMABaseTrainer, GDTrainer):
    pass