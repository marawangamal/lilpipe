"""NPO with a frozen base-model reference and WikiText retention."""

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

from configs.unlearn.utils import BalancedSourceSampler

# Match the released WMDP NPO baseline. Axolotl drops unknown YAML keys.
BETA = 0.0225
GAMMA = 1.0


class BalancedNPOTrainer(AxolotlTrainer):
    """Use the disabled LoRA adapter as the original model reference."""

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
            raise ValueError("NPO requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}

        forget_outputs = model(**forget_inputs)
        current_forget_ce = forget_outputs.loss.float()
        with torch.no_grad(), model.disable_adapter():
            reference_forget_ce = model(**forget_inputs).loss.float()
        retain_ce = model(**retain_inputs).loss.float()

        forget_loss = -(2 / BETA) * F.logsigmoid(
            BETA * (current_forget_ce - reference_forget_ce)
        )
        loss = forget_loss + GAMMA * retain_ce
        return (loss, forget_outputs) if return_outputs else loss
