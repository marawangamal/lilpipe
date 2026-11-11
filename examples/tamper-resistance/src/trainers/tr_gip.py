"""GIP defense and relearning training steps."""

import math

import torch
from axolotl.core.trainers.base import AxolotlTrainer

from trainers._base_unlearning import UnlearningTrainer
from trainers.ul_gd import GradDiffTrainer
from trainers.ul_npo import NPOTrainer
from trainers.tr_sam import NPOSAMTrainer
from trainers.utils import (
    BalancedSampler,
    accumulate_gradients,
    dict_binop,
    dict_dot,
    gradient_dict_dot,
    model_to_params,
    named_gradients,
)


class GIPTrainer(UnlearningTrainer):
    def __init__(
        self,
        *args,
        gip_mode="cosim_of_avgs",
        coeff_forget=0.0,
        coeff_retain=1.0,
        coeff_gip=0.01,
        eps_finite_diff=1e-3,
        eps_norm_div=1e-8,
        n_forget=5_000,
        n_retain=5_000,
        shuffle_streams=False,
        **kwargs,
    ):
        if gip_mode not in {"avg_of_cosims", "cosim_of_avgs"}:
            raise ValueError("gip_mode must be 'avg_of_cosims' or 'cosim_of_avgs'")
        if any(
            not math.isfinite(c) or c < 0
            for c in (coeff_forget, coeff_retain, coeff_gip)
        ):
            raise ValueError("Coefficients must be finite and nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in (eps_finite_diff, eps_norm_div)):
            raise ValueError(
                "eps_finite_diff and eps_norm_div must be finite and positive"
            )
        if any(type(n) is not int or n <= 0 for n in (n_forget, n_retain)):
            raise ValueError("Stream lengths must be positive integers")
        super().__init__(
            *args, coeff_forget=coeff_forget, coeff_retain=coeff_retain, **kwargs
        )
        self.gip_mode = gip_mode
        self.coeff_gip = coeff_gip
        self.eps_finite_diff = eps_finite_diff
        self.eps_norm_div = eps_norm_div
        self.n_forget = n_forget
        self.n_retain = n_retain
        self.shuffle_streams = shuffle_streams

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the contiguous forget and retain streams."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 4 or self._train_batch_size % 2:
            raise ValueError("GIP requires an even batch size of at least four")
        if len(dataset) < self.n_forget + self.n_retain:
            raise ValueError("GIP dataset is shorter than its configured streams")
        return BalancedSampler(
            self.n_forget,
            self.n_retain,
            shuffle=self.shuffle_streams,
            batch_size=self._train_batch_size,
            seed=(
                self.args.data_seed
                if self.args.data_seed is not None
                else self.args.seed
            ),
        )

    @staticmethod
    def compute_ce(model, batch):
        """Compute CE using the completion masks already stored in labels."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(**batch, use_cache=False).loss

    def compute_gip(self, model, forget, compute_loss_only=False):
        compute = {
            "avg_of_cosims": self.compute_gip_avg_of_cosims,
            "cosim_of_avgs": self.compute_gip_cosim_of_avgs,
        }[self.gip_mode]
        return compute(model, forget, compute_loss_only=compute_loss_only)

    def compute_gip_avg_of_cosims(self, model, forget, compute_loss_only=False):
        """Average cosine over all distinct pairs of per-example CE gradients."""
        B = len(forget["labels"])
        if B < 2:
            raise ValueError("GIP requires at least two forget examples")
        theta = model_to_params(model, only_trainable=True)

        # compute m, the mean normalized gradient
        m = {name: torch.zeros_like(p) for name, p in theta.items()}
        self_cosines = 0
        for i in range(B):
            batch = {name: value[i : i + 1] for name, value in forget.items()}
            g_theta_i = dict(
                zip(
                    theta,
                    torch.autograd.grad(
                        self.compute_ce(model, batch), tuple(theta.values())
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
                        self.compute_ce(model, batch), tuple(theta.values())
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

            theta_0 = {name: p.detach().clone() for name, p in theta.items()}
            try:
                with torch.no_grad():
                    for name, p in theta.items():
                        p.add_(self.eps_finite_diff * v_i[name])

                g_theta_i_plus_eps_v = dict(
                    zip(
                        theta,
                        torch.autograd.grad(
                            self.compute_ce(model, batch), tuple(theta.values())
                        ),
                    )
                )
            finally:
                with torch.no_grad():
                    for name, p in theta.items():
                        p.copy_(theta_0[name])

            difference = dict_binop(
                g_theta_i,
                g_theta_i_plus_eps_v,
                lambda original, shifted: (shifted - original) / B,
            )
            if g_hat is None:
                g_hat = difference
            else:
                g_hat = dict_binop(g_hat, difference, lambda a, b: a + b)

        scale = 2 * B / ((B - 1) * self.eps_finite_diff)
        return cosine, {name: scale * value for name, value in g_hat.items()}

    def compute_gip_cosim_of_avgs(self, model, forget, compute_loss_only=False):
        """Compute cosine between the mean CE gradients of two random forget groups."""
        B = len(forget["labels"])
        if B < 2 or B % 2:
            raise ValueError("GIP requires an even number of forget examples")
        theta = model_to_params(model, only_trainable=True)
        order = torch.randperm(B, device=forget["labels"].device)
        indices_a, indices_b = order.chunk(2)
        batch_a = {name: value[indices_a] for name, value in forget.items()}
        batch_b = {name: value[indices_b] for name, value in forget.items()}

        g_theta_a = dict(
            zip(
                theta,
                torch.autograd.grad(
                    self.compute_ce(model, batch_a), tuple(theta.values())
                ),
            )
        )
        g_theta_b = dict(
            zip(
                theta,
                torch.autograd.grad(
                    self.compute_ce(model, batch_b), tuple(theta.values())
                ),
            )
        )
        norm_a = dict_dot(g_theta_a, g_theta_a).sqrt()
        norm_b = dict_dot(g_theta_b, g_theta_b).sqrt()
        denom_a = norm_a + self.eps_norm_div
        denom_b = norm_b + self.eps_norm_div
        cosine = dict_dot(g_theta_a, g_theta_b) / (denom_a * denom_b)
        if compute_loss_only:
            return cosine

        # estimate gradient from the two groups
        g_hat = None
        for batch, g_theta_i, g_theta_j, norm_i, denom_i in (
            (batch_a, g_theta_a, g_theta_b, norm_a, denom_a),
            (batch_b, g_theta_b, g_theta_a, norm_b, denom_b),
        ):
            # compute v_i
            norm_i_safe = norm_i.clamp_min(torch.finfo(norm_i.dtype).tiny)
            v_i = {
                name: g_theta_j[name] / (denom_a * denom_b).to(p.device)
                - cosine.to(p.device)
                * (g_theta_i[name] / norm_i_safe.to(p.device))
                / denom_i.to(p.device)
                for name, p in theta.items()
            }

            theta_0 = {name: p.detach().clone() for name, p in theta.items()}
            try:
                with torch.no_grad():
                    for name, p in theta.items():
                        p.add_(self.eps_finite_diff * v_i[name])

                g_theta_i_plus_eps_v = dict(
                    zip(
                        theta,
                        torch.autograd.grad(
                            self.compute_ce(model, batch), tuple(theta.values())
                        ),
                    )
                )
            finally:
                with torch.no_grad():
                    for name, p in theta.items():
                        p.copy_(theta_0[name])

            difference = dict_binop(
                g_theta_i,
                g_theta_i_plus_eps_v,
                lambda original, shifted: shifted - original,
            )
            if g_hat is None:
                g_hat = difference
            else:
                g_hat = dict_binop(g_hat, difference, lambda a, b: a + b)

        return cosine, {
            name: value / self.eps_finite_diff for name, value in g_hat.items()
        }

    def accumulate_forget_grad(
        self, model, forget, coefficient=1.0, compute_loss_only=False
    ):
        """Accumulate the configured unlearning objective for forget examples."""
        loss = self.compute_forget_loss(model, forget)
        if compute_loss_only:
            return coefficient * loss.detach()

        parameters = model_to_params(model, only_trainable=True)
        gradients = named_gradients(
            loss, parameters, self.current_gradient_accumulation_steps
        )
        norm = gradient_dict_dot(gradients, gradients).sqrt() * abs(coefficient)
        accumulate_gradients(parameters, gradients, coefficient)
        return coefficient * loss.detach(), norm.detach()

    def accumulate_retain_grad(
        self, model, retain, coefficient=1.0, compute_loss_only=False
    ):
        """Accumulate the inherited retain loss, weighted once by coefficient."""
        loss = self.compute_retain_loss(model, retain)
        if compute_loss_only:
            return coefficient * loss.detach()

        parameters = model_to_params(model, only_trainable=True)
        gradients = named_gradients(
            loss, parameters, self.current_gradient_accumulation_steps
        )
        norm = gradient_dict_dot(gradients, gradients).sqrt() * abs(coefficient)
        accumulate_gradients(parameters, gradients, coefficient)
        return coefficient * loss.detach(), norm.detach()

    def accumulate_gip_grad(
        self, model, forget, coefficient=1.0, compute_loss_only=False
    ):
        """Return the weighted GIP loss and norm, accumulating its gradient."""
        if compute_loss_only:
            return (
                coefficient
                * self.compute_gip(model, forget, compute_loss_only=True).detach()
            )

        # compute_gip returns the cosine and its unscaled named gradients.
        cosine, gradients = self.compute_gip(model, forget)
        parameters = model_to_params(model, only_trainable=True)
        scale = coefficient / self.current_gradient_accumulation_steps
        norm = gradient_dict_dot(gradients, gradients).sqrt() * abs(scale)
        accumulate_gradients(parameters, gradients, scale)
        return coefficient * cosine.detach(), norm.detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        inp_forget = {name: value[0::2] for name, value in inputs.items()}
        inp_retain = {name: value[1::2] for name, value in inputs.items()}
        divisor = self.current_gradient_accumulation_steps

        configs = [
            {
                "name": "forget",
                "accum_fn": self.accumulate_forget_grad,
                "kwargs": {
                    "model": model,
                    "forget": inp_forget,
                    "coefficient": self.coeff_forget,
                },
            },
            {
                "name": "retain",
                "accum_fn": self.accumulate_retain_grad,
                "kwargs": {
                    "model": model,
                    "retain": inp_retain,
                    "coefficient": self.coeff_retain,
                },
            },
            {
                "name": "gip",
                "accum_fn": self.accumulate_gip_grad,
                "kwargs": {
                    "model": model,
                    "forget": inp_forget,
                    "coefficient": self.coeff_gip,
                },
            },
        ]

        log_dict = {}
        losses = []
        for cfg in configs:
            loss, norm = cfg["accum_fn"](**cfg["kwargs"])
            log_dict[f"loss_{cfg['name']}"] = loss.item()
            log_dict[f"norm_{cfg['name']}"] = norm.item()
            losses.append(loss)

        self.log(log_dict)
        loss = sum(value.to(losses[0].device) for value in losses)
        return (loss / divisor).to(self.args.device)


class GIPGDTrainer(GIPTrainer, GradDiffTrainer):
    """Combine gradient-difference unlearning with the GIP update."""


class GIPNPOTrainer(GIPTrainer, NPOTrainer):
    """Combine the shared NPO objective with the GIP gradient update."""


class GIPNPOSAMTrainer(NPOSAMTrainer, GIPTrainer):
    """Use the positional GIP streams and passively log cosine during NPO+SAM."""

    def _get_train_sampler(self, train_dataset=None):
        return GIPTrainer._get_train_sampler(self, train_dataset)

    def _sam_training_step(self, model, inputs):
        inp_forget = {name: value[0::2] for name, value in inputs.items()}
        cosine = self.compute_gip(model, inp_forget, compute_loss_only=True)
        loss = super()._sam_training_step(model, inputs)
        self.log({"loss_gip": cosine.item(), "cosine_similarity": cosine.item()})
        return loss


class GIPRelearnTrainer(GIPTrainer):
    def _get_train_sampler(self, train_dataset=None):
        return AxolotlTrainer._get_train_sampler(self, train_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        inp_forget = inputs

        divisor = self.current_gradient_accumulation_steps
        loss_forget = self.compute_ce(model, inp_forget)
        loss_gip = self.accumulate_gip_grad(model, inp_forget, compute_loss_only=True)
        self.accelerator.backward(loss_forget / divisor)
        self.log(
            {
                "loss_forget": loss_forget.item(),
                "loss_gip": loss_gip.item(),
                "cosine_similarity": loss_gip.item(),
            }
        )
        return (loss_forget / divisor).to(self.args.device)
