"""Gradient-difference loss terms for unlearning."""

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.utils import BalancedSampler

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class GDTrainer(AxolotlTrainer):
    """Maximize forget cross-entropy and minimize retain cross-entropy."""

    def __init__(
        self,
        *args,
        source_id_forget=0,
        source_id_retain=1,
        coeff_retain=1.0,
        coeff_forget=1.0,
        **kwargs,
    ):
        if type(source_id_forget) is not int or type(source_id_retain) is not int:
            raise ValueError("source IDs must be integers")
        if source_id_forget == source_id_retain:
            raise ValueError("forget and retain source IDs must differ")
        super().__init__(*args, **kwargs)
        self.source_id_forget = source_id_forget
        self.source_id_retain = source_id_retain
        self.coeff_retain = coeff_retain
        self.coeff_forget = coeff_forget

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the configured forget and retain sources."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 2 or self._train_batch_size % 2:
            raise ValueError("GD sampling requires an even batch size")

        source_ids = dataset["source_id"]
        sample_ids_forget = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_forget
        ]
        sample_ids_retain = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_retain
        ]
        if not sample_ids_forget or not sample_ids_retain:
            raise ValueError("GD sampling requires both forget and retain examples")
        return BalancedSampler(sample_ids_forget, sample_ids_retain)

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
