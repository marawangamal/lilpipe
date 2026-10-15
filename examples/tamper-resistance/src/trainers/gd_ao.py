"""Gradient difference with activation orthogonality."""

from trainers.activation_orthogonality import ActivationOrthogonalityMixin
from trainers.gd import GradDiffTrainer


class GDAOTrainer(ActivationOrthogonalityMixin, GradDiffTrainer):
    """Add forget-set activation orthogonality to gradient difference."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["is_forget"].bool()
        if int(forget_mask.sum()) < 2:
            raise ValueError("GD+AO requires at least two forget rows per batch")

        loss_forget, forget_outputs = self.compute_forget_loss(
            model,
            inputs,
            return_outputs=True,
            model_kwargs={"output_hidden_states": True, "use_cache": False},
        )
        loss_retain, retain_outputs = self.compute_retain_loss(
            model, inputs, return_outputs=True
        )

        loss_activation_orthogonality = self.compute_activation_orthogonality_loss(
            forget_outputs.hidden_states,
            inputs["attention_mask"][forget_mask],
        )
        loss = loss_forget + loss_retain + loss_activation_orthogonality
        outputs = {
            "loss_forget": forget_outputs.loss.float(),
            "loss_retain": retain_outputs.loss.float(),
            "loss_activation_orthogonality": loss_activation_orthogonality,
        }
        return (loss, outputs) if return_outputs else loss
