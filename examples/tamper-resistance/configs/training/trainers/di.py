"""Retain cross-entropy plus pairwise-orthogonal forget activations."""

from typing import List

import torch
from axolotl.core.trainers.base import AxolotlTrainer

from configs.training.trainers.samplers import MixedSourceSampler


def compute_mean_cosim(z: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """Compute mean cosine similarity between token-averaged activations.

    Args:
        z: Layer-wise activation tensors. Shape: (B, T, D) x L
        mask: Include mask. Shape (B, T)

    Returns:
        Mean cosine similarity. Shape (1,)
    """
    # compute layer-wise token means (excluding masked tokens)
    lengths = mask.sum(1).clamp_min(1).to(z[0].dtype)  # Shape: (B,)
    cosims = list()
    for l in range(len(z)):
        z_mean_l = (z[l] * mask.unsqueeze(-1)).sum(1) / lengths  # Shape: (B, D)
        norm_l = torch.linalg.norm(z_mean_l, keep_dims=True)
        cosim = z_mean_l @ z_mean_l.T / (norm_l * norm_l.T)  # Shape: (B, B)
        off_diag_mask = ~torch.eye(
            z_mean_l.shape[0], dtype=torch.bool, device=z_mean_l.device
        )
        cosims.append(cosim[off_diag_mask].mean())
    return torch.stack(cosims).mean()


class DITrainer(AxolotlTrainer):
    """Retain cross-entropy plus mean off-diagonal cosine of forget rows."""

    def __init__(
        self,
        *args,
        retain_coefficient=1.0,
        forget_coefficient=1.0,
        target_layers=(5, 10, 15, 20, 25, 30),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retain_coefficient = retain_coefficient
        self.forget_coefficient = forget_coefficient
        self.target_layers = target_layers
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

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute retain CE and forget-activation orthogonality loss.

        Args:
            model: Trainable causal language model.
            inputs: Batched tensors with keys:
                input_ids: Shape (B, T).
                attention_mask: Shape (B, T).
                labels: Shape (B, T).
                is_forget: Boolean tensor with shape (B,). True marks forget
                    rows; False marks retain rows. Requires at least one retain
                    and two forget rows.
            return_outputs: Whether to also return component losses.

        Note:
            expects batches with mixed samples some being forget, some being retain

        Returns:
            The loss, optionally paired with the component losses.
        """
        # NEW:
        sample_mask_forget = inputs["is_forget"].bool()
        sample_mask_retain = ~sample_mask_forget
        if not sample_mask_forget.any() or not sample_mask_retain.any():
            raise ValueError("DITrainer requires forget and retain rows in every batch")
        if int(sample_mask_forget.sum()) < 2:
            raise ValueError("DITrainer requires at least two forget rows per batch")

        # compute retain loss
        inputs_retain = {
            field: inputs[field][sample_mask_retain]
            for field in ("input_ids", "attention_mask", "labels")
        }
        loss_retain = model(**inputs_retain).loss

        # compute ortho loss
        attn_mask_forget = inputs["attention_mask"][sample_mask_forget]
        z_forget = model(
            input_ids=inputs["input_ids"][sample_mask_forget],
            attention_mask=attn_mask_forget,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states  # Shape: (B, T, D) x L
        loss_mean_cosim = compute_mean_cosim(z_forget, attn_mask_forget)

        loss = (
            self.retain_coefficient * loss_retain
            + self.forget_coefficient * loss_mean_cosim
        )

        return (
            (loss, {"loss_retain": loss_retain, "loss_mean_cosim": loss_mean_cosim})
            if return_outputs
            else loss
        )
