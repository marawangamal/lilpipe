"""Gradient difference with the DEFF data-inefficiency term."""

from trainers.deff import DataInefficiencyMixin
from trainers.gd import GradDiffTrainer


class GDDEFFTrainer(DataInefficiencyMixin, GradDiffTrainer):
    """Add forget-activation data inefficiency to gradient difference."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["is_forget"].bool()
        if int(forget_mask.sum()) < 2:
            raise ValueError("GD-DEFF requires at least two forget rows per batch")

        loss_forget, forget_outputs = self.compute_forget_loss(
            model,
            inputs,
            return_outputs=True,
            model_kwargs={"output_hidden_states": True, "use_cache": False},
        )
        loss_retain, retain_outputs = self.compute_retain_loss(
            model, inputs, return_outputs=True
        )

        loss_data_inefficiency = self.compute_data_inefficiency_loss(
            forget_outputs.hidden_states,
            inputs["attention_mask"][forget_mask],
        )
        loss = loss_forget + loss_retain + loss_data_inefficiency
        outputs = {
            "loss_forget": forget_outputs.loss.float(),
            "loss_retain": retain_outputs.loss.float(),
            "loss_data_inefficiency": loss_data_inefficiency,
        }
        return (loss, outputs) if return_outputs else loss
