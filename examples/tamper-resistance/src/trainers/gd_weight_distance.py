"""Gradient-difference unlearning with weight-distance regularization."""

from trainers.gd import GradDiffTrainer
from trainers.gradient_metrics import global_l2_norm
from trainers.weight_distance import WeightDistanceMixin


class GDWeightDistanceTrainer(WeightDistanceMixin, GradDiffTrainer):
    """Apply GradDiff while maximizing RMS distance from a reference."""

    def __init__(
        self,
        *args,
        reference_model_name_or_path,
        coefficient_forget=1.0,
        coefficient_retain=1.0,
        coefficient_weight_dist=1.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            forget_coefficient=coefficient_forget,
            retain_coefficient=coefficient_retain,
            coefficient_weight_dist=coefficient_weight_dist,
            reference_model_name_or_path=reference_model_name_or_path,
            **kwargs,
        )
        self._weight_distance_logged_step = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_retain, grad_retain, outputs_retain = self.compute_retain_loss(
            model, inputs, return_grads=True, return_outputs=True
        )
        loss_forget, grad_forget = self.compute_forget_loss(
            model, inputs, return_grads=True
        )
        loss_weight_dist, grad_weight_dist = self.compute_weight_dist_loss(
            return_grads=True
        )
        loss = loss_retain + loss_forget + loss_weight_dist

        step = self.state.global_step
        if step != self._weight_distance_logged_step:
            self.log(
                {
                    "loss_weight_dist": loss_weight_dist.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "loss_forget": loss_forget.detach().item(),
                    "grad_norm_weight_dist": global_l2_norm(
                        gradient
                        for gradient in grad_weight_dist
                        if gradient is not None
                    ).item(),
                    "grad_norm_retain": global_l2_norm(
                        gradient for gradient in grad_retain if gradient is not None
                    ).item(),
                    "grad_norm_forget": global_l2_norm(
                        gradient for gradient in grad_forget if gradient is not None
                    ).item(),
                }
            )
            self._weight_distance_logged_step = step

        return (loss, outputs_retain) if return_outputs else loss
