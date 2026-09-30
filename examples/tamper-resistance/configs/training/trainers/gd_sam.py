"""Gradient-difference unlearning with SAM on the forget objective."""

from configs.training.trainers.sam import SAMAttack, TARTrainer
from configs.training.trainers.samplers import MixedSourceSampler


class GradDiffTrainer(TARTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        rho=0.01,
        retain_coeff=1.0,
        forget_coeff=1.0,
        attack_type="sam",
        attack_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            rho=rho,
            retain_coeff=retain_coeff,
            forget_coeff=forget_coeff,
            attack_type=attack_type,
            attack_kwargs=attack_kwargs,
            **kwargs,
        )

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @staticmethod
    def split_sources(inputs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("GradDiff requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return forget_inputs, retain_inputs

    def compute_forget_loss(self, model_forward, inputs, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        outputs = model_forward(**forget_inputs)
        return -outputs.loss.float()

    def compute_retain_loss(self, model_forward, inputs, **kwargs):
        _, retain_inputs = self.split_sources(inputs)
        outputs = model_forward(**retain_inputs)
        return outputs.loss.float()
