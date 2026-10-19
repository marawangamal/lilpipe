"""Self-destructive models (SEAM): couple forget and retain gradients."""

import torch

from trainers.ul_gd import GradDiffTrainer
from trainers.ul_me import MaxEntTrainer


def grad_old(compute_loss, model, inputs, shift=None):
    """Gradient of ``compute_loss`` by parameter name, at the weights plus ``shift``."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    theta = {n: p.data for n, p in params.items()}
    for n, s in (shift or {}).items():
        params[n].data = params[n].data + s
    g = torch.autograd.grad(compute_loss(model, inputs), list(params.values()))
    for n, p in params.items():
        p.data = theta[n]
    return dict(zip(params, g))


def grad(compute_loss, model, inputs, shift=None):
    """Gradient of ``compute_loss`` by parameter name, at the weights plus ``shift``."""
    theta = {name: param.data for name, param in model.named_parameters()}
    # theta <- theta + delta
    for name, param in model.named_parameters():
        if shift is not None and name in shift:
            param.data = param.data + shift[name]

    loss = compute_loss(model, inputs)
    loss.backward()
    g = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    for name, param in model.named_parameters():
        param.data = theta[name]
    return g


def dot(a, b):
    """Dot product of two name-to-tensor dicts that may span several devices."""
    device = next(iter(a.values())).device
    return sum(torch.sum(a[n].float() * b[n].float()).to(device) for n in a)


class SEAMBaseTrainer:
    """Add a self-destructive gradient-coupling term to an unlearning trainer.

    Mix in before an unlearning trainer whose ``compute_forget_loss`` and
    ``compute_retain_loss`` accept a model-like callable and return a loss. The
    coupling term is the cosine similarity between the forget and retain
    gradients; minimizing it makes relearning the forget set degrade utility.
    """

    def __init__(self, *args, sd_coefficient=1.0, sd_epsilon=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_coefficient = sd_coefficient
        self.sd_epsilon = sd_epsilon

    def compute_attack_loss(self, model, inputs):
        """Cross-entropy on the forget rows, as minimized by a relearning attack."""
        forget_mask = inputs["is_forget"].bool()
        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        return self.compute_loss_term(model, forget_inputs, 1.0)

    def compute_sd_loss(self, model, inputs, epsilon=1e-3):
        """Cosine similarity between forget and retain gradients.

        TODO: this differentiates through both gradients (double backward). SEAM
        uses a Hessian-free finite-difference estimate instead; swap it in here.
        """
        theta = {n: p for n, p in model.named_parameters() if p.requires_grad}

        # Gradients at current weights
        g_f = grad(self.compute_attack_loss, model, inputs)  # attacker minimizes -l_f
        g_r = grad(self.compute_retain_loss, model, inputs)

        # Normalize
        norm_f, norm_r = dot(g_f, g_f).sqrt().item(), dot(g_r, g_r).sqrt().item()
        c = dot(g_f, g_r) / (norm_f * norm_r)
        tilde_g_f = {n: g / norm_f for n, g in g_f.items()}
        tilde_g_r = {n: g / norm_r for n, g in g_r.items()}

        # Perturbation directions
        d_f = {n: epsilon * (tilde_g_r[n] - c.item() * tilde_g_f[n]) for n in g_f}
        d_r = {n: epsilon * (tilde_g_f[n] - c.item() * tilde_g_r[n]) for n in g_f}

        # Gradients at temporary perturbed weights; same batches
        g_f_shifted = grad(self.compute_attack_loss, model, inputs, d_f)
        g_r_shifted = grad(self.compute_retain_loss, model, inputs, d_r)

        # Approximate gradient of cos(g_f, g_r)
        g_sd = {
            n: (g_f_shifted[n] - g_f[n]) / (epsilon * norm_f)
            + (g_r_shifted[n] - g_r[n]) / (epsilon * norm_r)
            for n in g_f
        }

        # 2. Inject that estimate through a surrogate loss
        # its value is c, its gradient is g_sd
        loss_sd_surrogate = dot(theta, g_sd)
        loss_sd_surrogate = loss_sd_surrogate - loss_sd_surrogate.detach() + c

        return loss_sd_surrogate

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_forget = self.compute_forget_loss(model, inputs)
        loss_retain = self.compute_retain_loss(model, inputs)
        loss_sd = self.compute_sd_loss(model, inputs, self.sd_epsilon)
        loss_sd = loss_sd.to(loss_forget.device)
        loss = loss_forget + loss_retain + self.sd_coefficient * loss_sd

        if model.training and torch.is_grad_enabled():
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "loss_coupling": loss_sd.detach().item(),
                }
            )
        return (loss, None) if return_outputs else loss


class SEAMGDTrainer(SEAMBaseTrainer, GradDiffTrainer):
    """SEAM with gradient-difference losses."""


class SEAMMETrainer(SEAMBaseTrainer, MaxEntTrainer):
    """SEAM with MaxEnt losses."""
