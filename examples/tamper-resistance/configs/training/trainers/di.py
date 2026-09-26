"""Retain cross-entropy plus pairwise-orthogonal forget activations."""

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

from configs.training.trainers.samplers import MixedSourceSampler

TARGET_LAYERS = (5, 10, 15, 20, 25, 30)


def masked_pool(activations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean each row over its valid tokens, shape (batch, hidden)."""
    lengths = mask.sum(dim=1).clamp_min(1).to(activations.dtype)
    return (activations * mask[:, :, None]).sum(dim=1) / lengths[:, None]


class DITrainer(AxolotlTrainer):
    """Retain cross-entropy plus mean off-diagonal cosine of forget rows."""

    forget_coefficient = 1.0
    retain_coefficient = 1.0

    def __init__(self, *args, target_layers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers or TARGET_LAYERS)
        num_layers = self.model.config.num_hidden_layers
        if len(set(self.target_layers)) != len(self.target_layers) or any(
            layer < 0 or layer >= num_layers for layer in self.target_layers
        ):
            raise ValueError(
                f"target_layers must be unique indices in 0-{num_layers - 1}"
            )

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        source = inputs["cb_source"]
        forget_mask = source == 1
        retain_mask = source == 0
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("DITrainer requires forget and retain rows in every batch")
        if int(forget_mask.sum()) < 2:
            raise ValueError("DITrainer requires at least two forget rows per batch")

        fields = ("input_ids", "attention_mask", "labels")
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        retain_ce = model(**retain_inputs).loss.float()

        forget_attention = inputs["attention_mask"][forget_mask]
        hidden_states = model(
            input_ids=inputs["input_ids"][forget_mask],
            attention_mask=forget_attention,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states

        cosines = []
        for layer in self.target_layers:
            pooled = F.normalize(
                masked_pool(hidden_states[layer + 1].float(), forget_attention),
                dim=-1,
            )
            similarities = pooled @ pooled.transpose(0, 1)
            off_diagonal = ~torch.eye(
                pooled.shape[0], dtype=torch.bool, device=pooled.device
            )
            cosines.append(similarities[off_diagonal].mean())
        loss_orth = torch.stack(cosines).mean()

        loss = self.retain_coefficient * retain_ce + self.forget_coefficient * loss_orth
        self.log(
            {
                "loss_retain": retain_ce.detach().item(),
                "loss_orth": loss_orth.detach().item(),
                "retain_coefficient": self.retain_coefficient,
                "forget_coefficient": self.forget_coefficient,
                "retain_count": int(retain_mask.sum()),
                "forget_count": int(forget_mask.sum()),
            }
        )
        return (
            (loss, {"retain_ce": retain_ce, "orth_cosine": loss_orth})
            if return_outputs
            else loss
        )
