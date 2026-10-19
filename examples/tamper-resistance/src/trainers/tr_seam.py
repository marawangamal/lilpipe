"""Self-destructive models (SEAM): couple forget and retain gradients."""

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from trainers.ul_gd import GradDiffTrainer
from trainers.ul_me import MaxEntTrainer


def grad(compute_loss, model, inputs, shift=0):
    """Flat gradient of ``compute_loss`` at the trainable weights plus ``shift``."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    theta = parameters_to_vector(parameters).detach()
    vector_to_parameters(theta + shift, parameters)
    g = torch.autograd.grad(compute_loss(model, inputs), parameters)
    vector_to_parameters(theta, parameters)
    return parameters_to_vector(g)


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
        theta = parameters_to_vector(p for p in model.parameters() if p.requires_grad)

        # Gradients at current weights
        g_f = grad(self.compute_attack_loss, model, inputs)  # attacker minimizes -l_f
        g_r = grad(self.compute_retain_loss, model, inputs)

        # Normalize
        norm_f, norm_r = g_f.norm(), g_r.norm()
        tilde_g_f = g_f / norm_f
        tilde_g_r = g_r / norm_r
        c = torch.dot(tilde_g_f, tilde_g_r)

        # Perturbation directions
        d_f = tilde_g_r - c * tilde_g_f
        d_r = tilde_g_f - c * tilde_g_r

        # Gradients at temporary perturbed weights; same batches
        g_f_shifted = grad(self.compute_attack_loss, model, inputs, epsilon * d_f)
        g_r_shifted = grad(self.compute_retain_loss, model, inputs, epsilon * d_r)

        # Approximate gradient of cos(g_f, g_r)
        g_sd = (g_f_shifted - g_f) / (epsilon * norm_f) + (g_r_shifted - g_r) / (
            epsilon * norm_r
        )

        # 2. Inject that estimate through a surrogate loss
        # its value is c, its gradient is g_sd
        loss_sd_surrogate = torch.dot(theta, g_sd.detach())
        loss_sd_surrogate = loss_sd_surrogate - loss_sd_surrogate.detach() + c

        return loss_sd_surrogate

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_forget = self.compute_forget_loss(model, inputs)
        loss_retain = self.compute_retain_loss(model, inputs)
        loss_sd = self.compute_sd_loss(model, inputs, self.sd_epsilon)
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
