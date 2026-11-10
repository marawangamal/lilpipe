"""GIP defense and relearning training steps.

Coefficient initialization and sampling setup are still pending.
"""

import torch

from trainers.rl_bf16 import BF16Trainer
from trainers.tr_seam import (
    accumulate_gradients,
    gradient_dict_dot,
    model_to_params,
    named_gradients,
)

from trainers.utils import dict_binop, dict_dot


class GIPTrainer(BF16Trainer):
    def __init__(self, *args, gip_mode="cosim_of_avgs", **kwargs):
        if gip_mode not in {"avg_of_cosims", "cosim_of_avgs"}:
            raise ValueError("gip_mode must be 'avg_of_cosims' or 'cosim_of_avgs'")
        super().__init__(*args, **kwargs)
        self.gip_mode = gip_mode

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

        # compute m = \sum_i g_i / \|g_i\|
        m = {name: torch.zeros_like(p) for name, p in theta.items()}
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
            for name, p in theta.items():
                m[name] += g_theta_i[name] / (B * norm_i.to(p.device))

        cosine = (B * dict_dot(m, m) - 1) / (B - 1)
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
            u_i = {
                name: value / norm_i.to(value.device)
                for name, value in g_theta_i.items()
            }
            projection = dict_dot(m, u_i)
            v_i = {
                name: (m[name] - projection.to(p.device) * u_i[name])
                / norm_i.to(p.device)
                for name, p in theta.items()
            }

            theta_0 = {name: p.detach().clone() for name, p in theta.items()}
            try:
                with torch.no_grad():
                    for name, p in theta.items():
                        p.add_(self.eps * v_i[name])

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

        scale = 2 * B / ((B - 1) * self.eps)
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
        norm_a = (dict_dot(g_theta_a, g_theta_a) + self.cosine_tau**2).sqrt()
        norm_b = (dict_dot(g_theta_b, g_theta_b) + self.cosine_tau**2).sqrt()
        cosine = dict_dot(g_theta_a, g_theta_b) / (norm_a * norm_b)
        if compute_loss_only:
            return cosine

        # estimate gradient from the two groups
        g_hat = None
        for batch, g_theta_i, g_theta_j, norm_i in (
            (batch_a, g_theta_a, g_theta_b, norm_a),
            (batch_b, g_theta_b, g_theta_a, norm_b),
        ):
            # compute v_i
            v_i = {
                name: g_theta_j[name] / (norm_a * norm_b).to(p.device)
                - cosine.to(p.device) * g_theta_i[name] / norm_i.square().to(p.device)
                for name, p in theta.items()
            }

            theta_0 = {name: p.detach().clone() for name, p in theta.items()}
            try:
                with torch.no_grad():
                    for name, p in theta.items():
                        p.add_(self.eps * v_i[name])

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

        return cosine, {name: value / self.eps for name, value in g_hat.items()}

    def accumulate_forget_grad(
        self, model, forget, coefficient=1.0, compute_loss_only=False
    ):
        """Accumulate negative CE for forget examples."""
        loss = -self.compute_ce(model, forget)
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
        """Accumulate CE for retain examples."""
        loss = self.compute_ce(model, retain)
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
        """Accumulate weighted GIP gradients; return raw cosine and weighted norm."""
        if compute_loss_only:
            return self.compute_gip(model, forget, compute_loss_only=True).detach()

        # compute_gip returns the cosine and its unscaled named gradients.
        cosine, gradients = self.compute_gip(model, forget)
        parameters = model_to_params(model, only_trainable=True)
        scale = coefficient / self.current_gradient_accumulation_steps
        norm = gradient_dict_dot(gradients, gradients).sqrt() * abs(scale)
        accumulate_gradients(parameters, gradients, scale)
        return cosine.detach(), norm.detach()

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
            losses.append(self.coeff_gip * loss if cfg["name"] == "gip" else loss)

        self.log(log_dict)
        loss = sum(value.to(losses[0].device) for value in losses)
        return (loss / divisor).to(self.args.device)


class GIPRelearnTrainer(GIPTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        inp_forget = {name: value[0::2] for name, value in inputs.items()}

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
