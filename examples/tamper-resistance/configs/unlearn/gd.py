"""Plain gradient-difference unlearning with balanced forget and retain batches."""

from axolotl.core.trainers.base import AxolotlTrainer

from configs.unlearn.utils import BalancedSourceSampler


class BalancedGradDiffTrainer(AxolotlTrainer):
    """Minimize retain cross-entropy while maximizing forget cross-entropy."""

    forget_coefficient = 1.0
    retain_coefficient = 1.0

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
        loss = (
            -self.forget_coefficient * forget_ce + self.retain_coefficient * retain_ce
        )
        return (loss, forget_outputs) if return_outputs else loss


class BalancedGradDiffF01R1Trainer(BalancedGradDiffTrainer):
    """Reduce forget pressure tenfold while keeping retain pressure fixed."""

    forget_coefficient = 0.1


class BalancedGradDiffF01R05Trainer(BalancedGradDiffF01R1Trainer):
    """Try a smaller retain weight against the same forget pressure."""

    retain_coefficient = 0.5


class BalancedGradDiffF01R15Trainer(BalancedGradDiffF01R1Trainer):
    """Try an intermediate retain weight of 1.5."""

    retain_coefficient = 1.5


class BalancedGradDiffF01R2Trainer(BalancedGradDiffF01R1Trainer):
    """Try an intermediate retain weight of 2.0."""

    retain_coefficient = 2.0


class BalancedGradDiffF01R4Trainer(BalancedGradDiffF01R1Trainer):
    """Use four times the retain pressure of the 0.1/1.0 variant."""

    retain_coefficient = 4.0
