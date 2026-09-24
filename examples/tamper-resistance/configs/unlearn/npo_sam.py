"""NPO with a LoRA-space SAM pass on the forget loss only."""

import torch

from configs.unlearn.npo import BETA, GAMMA, BalancedNPOTrainer

RHO = 0.01


def sam_perturbation(gradients, rho=RHO):
    """Return a radius-rho ascent step for a dictionary of gradients."""
    active = [gradient for gradient in gradients.values() if gradient is not None]
    if not active:
        return {}
    norm = torch.linalg.vector_norm(
        torch.stack([gradient.detach().float().norm() for gradient in active])
    )
    if not norm.isfinite() or norm == 0:
        return {}
    return {
        name: rho * gradient.detach() / norm
        for name, gradient in gradients.items()
        if gradient is not None
    }


class BalancedNPOSAMTrainer(BalancedNPOTrainer):
    """Accumulate perturbed NPO forget and ordinary WikiText retain gradients."""

    beta = BETA
    gamma = GAMMA
    rho = RHO

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, inputs)

    def _gradient(self, loss_fn, theta):
        """Backward one loss, then take its gradients out of Axolotl's buffers."""
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in theta.items()}
        for parameter in theta.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        x_forget, x_retain = self.split_sources(self._prepare_inputs(inputs))
        theta = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        accumulated_grad = {name: parameter.grad for name, parameter in theta.items()}
        for parameter in theta.values():
            parameter.grad = None

        try:
            _, g = self._gradient(
                lambda: self.forget_loss(model, x_forget, beta=self.beta)[0], theta
            )
            delta = sam_perturbation(g, self.rho)

            original_theta = {}
            try:
                with torch.no_grad():
                    for name, step in delta.items():
                        original_theta[name] = theta[name].detach().clone()
                        theta[name].add_(step)
                forget_loss, g_forget_prime = self._gradient(
                    lambda: self.forget_loss(model, x_forget, beta=self.beta)[0], theta
                )
            finally:
                with torch.no_grad():
                    for name, original in original_theta.items():
                        theta[name].copy_(original)

            retain_loss, g_retain = self._gradient(
                lambda: self.gamma * model(**x_retain).loss.float(), theta
            )
        finally:
            for name, parameter in theta.items():
                parameter.grad = accumulated_grad[name]

        for name, parameter in theta.items():
            for gradient in (g_forget_prime[name], g_retain[name]):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )

        return (
            (forget_loss + retain_loss) / self.current_gradient_accumulation_steps
        ).detach()


class BalancedNPOSAMRho003Trainer(BalancedNPOSAMTrainer):
    rho = 0.003


class BalancedNPOSAMGamma225Trainer(BalancedNPOSAMTrainer):
    gamma = 2.25


class BalancedNPOSAMGamma450Trainer(BalancedNPOSAMTrainer):
    gamma = 4.5


class BalancedNPOSAMBeta015Gamma225Trainer(BalancedNPOSAMGamma225Trainer):
    beta = 0.015
