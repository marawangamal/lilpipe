"""Plain gradient-difference unlearning with balanced forget and retain batches."""

from axolotl.core.trainers.base import AxolotlTrainer

from configs.unlearn.utils import BalancedSourceSampler


class BalancedGradDiffTrainer(AxolotlTrainer):
    """Minimize retain cross-entropy while maximizing forget cross-entropy."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return BalancedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        source = inputs["cb_source"]
        forget_mask = source == 1
        retain_mask = source == 0
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("GradDiff requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}

        forget_outputs = model(**forget_inputs)
        forget_ce = forget_outputs.loss.float()
        retain_ce = model(**retain_inputs).loss.float()
        loss = -forget_ce + retain_ce
        return (loss, forget_outputs) if return_outputs else loss
