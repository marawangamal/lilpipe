"""Gradient-difference unlearning with weight-distance regularization."""

from trainers.gd import GradDiffTrainer
from trainers.weight_distance import global_rms_distance


# TODO: use a WeightDistanceMixin later on that exposes
# get_dist_from_ref()
class GDWeightDistanceTrainer(GradDiffTrainer):
    """Apply GradDiff while maximizing RMS distance from the initial weights."""

    def __init__(
        self,
        *args,
        coefficient_forget=1.0,
        coefficient_retain=1.0,
        coefficient_weight_dist=1.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            forget_coefficient=coefficient_forget,
            retain_coefficient=coefficient_retain,
            **kwargs,
        )
        self.coefficient_weight_dist = coefficient_weight_dist
        self._theta_ref = None
        self._weight_distance_logged_step = None

    @property
    def theta(self):
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    @property
    def theta_ref(self):
        theta = self.theta
        if self._theta_ref is None:
            self._theta_ref = {
                name: parameter.detach().clone() for name, parameter in theta.items()
            }
        if theta.keys() != self._theta_ref.keys():
            raise RuntimeError(
                "trainable parameters changed after weight-distance snapshot"
            )
        return self._theta_ref

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        theta = self.theta
        theta_ref = self.theta_ref

        loss_gd, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        weight_distance = global_rms_distance(theta.values(), theta_ref.values())
        loss_wd = -self.coefficient_weight_dist * weight_distance
        loss = loss_gd + loss_wd

        step = self.state.global_step
        if step != self._weight_distance_logged_step:
            self.log(
                {
                    "loss_wd": loss_wd.detach().item(),
                    "loss_gd": loss_gd.detach().item(),
                }
            )
            self._weight_distance_logged_step = step

        return (loss, outputs) if return_outputs else loss
