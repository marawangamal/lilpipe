"""Gradient-difference unlearning with weight-distance regularization."""

from trainers.gd import GradDiffTrainer
from trainers.weight_distance import global_l2_distance


class GDWeightDistTrainer(GradDiffTrainer):
    """Apply GradDiff while maximizing distance from the initial weights."""

    def __init__(self, *args, distance_coefficient=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_coefficient = distance_coefficient
        self._weight_dist_anchors = None
        self._weight_dist_logged_step = None

    def _parameters_and_anchors(self, model):
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if self._weight_dist_anchors is None:
            self._weight_dist_anchors = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
        if parameters.keys() != self._weight_dist_anchors.keys():
            raise RuntimeError(
                "trainable parameters changed after weight-distance snapshot"
            )
        return parameters, self._weight_dist_anchors

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        parameters, anchors = self._parameters_and_anchors(model)
        grad_diff_loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        distance = global_l2_distance(parameters.values(), anchors.values())
        step = self.state.global_step
        if step != self._weight_dist_logged_step:
            self.log({"weight_distance": distance.detach().item()})
            self._weight_dist_logged_step = step
        loss = grad_diff_loss - self.distance_coefficient * distance
        return (loss, outputs) if return_outputs else loss
