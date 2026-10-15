"""Data-inefficiency loss over forget-set activations."""

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.sampling import MixedSourceSampler

DEFAULT_TARGET_LAYERS = (5, 10, 15, 20, 25, 30)


def mean_pairwise_cosine(hidden_states, attention_mask):
    """Average off-diagonal cosine of masked-mean activations across layers."""
    lengths = attention_mask.sum(1).clamp_min(1).to(hidden_states[0].dtype)
    off_diagonal_cosines = []
    for layer_hidden_states in hidden_states:
        pooled = (layer_hidden_states * attention_mask.unsqueeze(-1)).sum(
            1
        ) / lengths.unsqueeze(-1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        pairwise_cosine = pooled @ pooled.T
        off_diagonal = ~torch.eye(
            pooled.shape[0], dtype=torch.bool, device=pooled.device
        )
        off_diagonal_cosines.append(pairwise_cosine[off_diagonal].mean())
    return torch.stack(off_diagonal_cosines).mean()


def validate_target_layers(target_layers, num_layers):
    """Return validated zero-based transformer block indices."""
    target_layers = tuple(target_layers)
    if (
        not target_layers
        or len(set(target_layers)) != len(target_layers)
        or any(layer < 0 or layer >= num_layers for layer in target_layers)
    ):
        raise ValueError(f"target_layers must be unique indices in 0-{num_layers - 1}")
    return target_layers


class DataInefficiencyMixin:
    """Provide the forget-activation data-inefficiency term and sampler."""

    def __init__(
        self,
        *args,
        data_inefficiency_coefficient=1.0,
        target_layers=None,
        **kwargs,
    ):
        if data_inefficiency_coefficient < 0:
            raise ValueError("data_inefficiency_coefficient must be non-negative")
        super().__init__(*args, **kwargs)
        self.data_inefficiency_coefficient = data_inefficiency_coefficient
        if target_layers is None:
            target_layers = range(self.model.config.num_hidden_layers)
        self.target_layers = validate_target_layers(
            target_layers, self.model.config.num_hidden_layers
        )

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    def compute_data_inefficiency_loss(self, hidden_states, attention_mask):
        selected = tuple(hidden_states[layer + 1] for layer in self.target_layers)
        return self.data_inefficiency_coefficient * mean_pairwise_cosine(
            selected, attention_mask
        )


class DEFFTrainer(DataInefficiencyMixin, AxolotlTrainer):
    """Minimize retain CE and forget-set data efficiency."""

    def __init__(
        self,
        *args,
        retain_coefficient=1.0,
        target_layers=DEFAULT_TARGET_LAYERS,
        **kwargs,
    ):
        super().__init__(*args, target_layers=target_layers, **kwargs)
        self.retain_coefficient = retain_coefficient

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("DEFF requires forget and retain rows in every batch")
        if int(forget_mask.sum()) < 2:
            raise ValueError("DEFF requires at least two forget rows per batch")

        fields = ("input_ids", "attention_mask", "labels")
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        loss_retain = model(**retain_inputs).loss.float()

        forget_attention = inputs["attention_mask"][forget_mask]
        hidden_states = model(
            input_ids=inputs["input_ids"][forget_mask],
            attention_mask=forget_attention,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        loss_data_inefficiency = self.compute_data_inefficiency_loss(
            hidden_states, forget_attention
        )

        loss = self.retain_coefficient * loss_retain + loss_data_inefficiency
        outputs = {
            "loss_retain": loss_retain,
            "loss_data_inefficiency": loss_data_inefficiency,
        }
        return (loss, outputs) if return_outputs else loss
