"""Gradient-difference loss terms for unlearning."""

from axolotl.core.trainers.base import AxolotlTrainer

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class GDTrainer(AxolotlTrainer):
    """Maximize forget cross-entropy and minimize retain cross-entropy."""

    def compute_loss_forget(self, model, inputs):
        return -model(**inputs).loss.float()

    def compute_loss_retain(self, model, inputs):
        return model(**inputs).loss.float()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["source_id"] == self.source_id_forget
        retain_mask = inputs["source_id"] == self.source_id_retain

        x_forget = {field: inputs[field][forget_mask] for field in INP_FIELDS}
        x_retain = {field: inputs[field][retain_mask] for field in INP_FIELDS}

        loss_forget = self.compute_loss_forget(model, x_forget)
        loss_retain = self.compute_loss_retain(model, x_retain)
        loss = self.coeff_forget * loss_forget + self.coeff_retain * loss_retain
        return (loss, None) if return_outputs else loss
