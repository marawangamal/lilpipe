"""NPO with a frozen base-model reference and WikiText retention."""

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer
from transformers import AutoModelForCausalLM

from configs.training.trainers.samplers import MixedSourceSampler

# Match the released WMDP NPO baseline. Axolotl drops unknown YAML keys.
BETA = 0.0225
GAMMA = 1.0


class NPOTrainer(AxolotlTrainer):
    """Use the disabled LoRA adapter as the original model reference."""

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @staticmethod
    def split_sources(inputs):
        source = inputs["cb_source"]
        forget_mask = source == 1
        retain_mask = source == 0
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("NPO requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return forget_inputs, retain_inputs

    @staticmethod
    def forget_loss(model, forget_inputs, beta=BETA):
        forget_outputs = model(**forget_inputs)
        current_forget_ce = forget_outputs.loss.float()
        with torch.no_grad(), model.disable_adapter():
            reference_forget_ce = model(**forget_inputs).loss.float()
        forget_loss = -(2 / beta) * F.logsigmoid(
            beta * (current_forget_ce - reference_forget_ce)
        )
        return forget_loss, forget_outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs, retain_inputs = self.split_sources(inputs)
        forget_loss, forget_outputs = self.forget_loss(model, forget_inputs)
        retain_ce = model(**retain_inputs).loss.float()
        loss = forget_loss + GAMMA * retain_ce
        return (loss, forget_outputs) if return_outputs else loss


def sequence_cross_entropy(logits, labels):
    """Return each document's mean next-token cross-entropy."""

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, reduction="none"
    )
    valid = shifted_labels != -100
    return (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class FullModelNPOTrainer(NPOTrainer):
    """NPO against a separate frozen Zephyr reference, plus retain CE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = None

    def forget_loss(self, model, forget_inputs):
        if self.reference_model is None:
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                "HuggingFaceH4/zephyr-7b-beta",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
            self.reference_model.to(next(model.parameters()).device)

        forget_outputs = model(**forget_inputs)
        current_ce = sequence_cross_entropy(
            forget_outputs.logits, forget_inputs["labels"]
        )
        with torch.no_grad():
            reference_logits = self.reference_model(**forget_inputs).logits
            reference_ce = sequence_cross_entropy(
                reference_logits, forget_inputs["labels"]
            )
        loss = -(2 / BETA) * F.logsigmoid(BETA * (current_ce - reference_ce)).mean()
        return loss, forget_outputs
