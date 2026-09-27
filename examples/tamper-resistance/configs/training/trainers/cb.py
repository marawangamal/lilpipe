"""Orthogonal circuit-breaker objective for WMDP unlearning."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

from configs.training.trainers.samplers import MixedSourceSampler


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype).unsqueeze(0).expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1)


def orthogonalization_loss(activations: torch.Tensor, mask: torch.Tensor):
    """Penalize positive cosine between different forget sequences."""

    lengths = mask.sum(dim=1).clamp_min(1).to(activations.dtype)
    pooled = (activations * mask[None, :, :, None]).sum(dim=2)
    pooled = F.normalize((pooled / lengths[None, :, None]).float(), dim=-1)
    similarities = pooled @ pooled.transpose(-1, -2)
    off_diagonal = ~torch.eye(
        activations.shape[1], dtype=torch.bool, device=activations.device
    )
    positive_pairs = F.relu(similarities[:, off_diagonal])
    if positive_pairs.numel() == 0:
        return activations.float().sum() * 0
    return positive_pairs.mean() * lengths.float().mean()


class CBTrainer(AxolotlTrainer):
    """Retain WikiText and reroute WMDP activations without token loss."""

    def __init__(
        self,
        *args,
        coeff_retain=10,
        coeff_forget=23,
        coeff_ortho=5,
        target_layers=(5, 10, 15, 20, 25, 30),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers)
        self.coeff_retain = coeff_retain
        self.coeff_forget = coeff_forget
        self.coeff_ortho = coeff_ortho
        self.microstep = 0
        self.total_microsteps = (
            self.args.max_steps * self.args.gradient_accumulation_steps
        )
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

    def get_coeffs(
        self,
        microstep: int,
        total_microsteps: int,
    ) -> tuple[float, float, float]:
        progress = min(max(microstep, 0) / max(total_microsteps - 1, 1), 1.0)
        return (
            self.coeff_retain * (0.1 + 0.9 * progress),
            self.coeff_forget * (1.0 - 0.25 * progress),
            self.coeff_ortho * progress,
        )

    def get_activations(
        self,
        model,
        input_ids,
        attention_mask,
        disable_adapter=False,
        disable_grad=False,
    ):
        adapter_context = model.disable_adapter() if disable_adapter else nullcontext()
        gradient_context = torch.no_grad() if disable_grad else nullcontext()
        was_training = model.training
        model.eval() if disable_grad else model.train()
        try:
            with adapter_context, gradient_context:
                hidden_states = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states
                result = torch.stack(
                    [hidden_states[layer + 1] for layer in self.target_layers]
                )
        finally:
            model.train(was_training)
        return result.detach() if disable_grad else result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_mask_forget = inputs["is_forget"].bool()
        sample_mask_retain = ~sample_mask_forget

        attn_mask_retain = inputs["attention_mask"][sample_mask_retain]
        attn_mask_forget = inputs["attention_mask"][sample_mask_forget]

        x_retain = inputs["input_ids"][sample_mask_retain]
        x_forget = inputs["input_ids"][sample_mask_forget]

        loss = 0
        z_retain_ref = self.get_activations(
            model,
            x_retain,
            attn_mask_retain,
            disable_adapter=True,
            disable_grad=True,
        )
        z_forget_ref = self.get_activations(
            model,
            x_forget,
            attn_mask_forget,
            disable_adapter=True,
            disable_grad=True,
        )
        z_retain = self.get_activations(
            model,
            x_retain,
            attn_mask_retain,
        )
        z_forget = self.get_activations(
            model,
            x_forget,
            attn_mask_forget,
        )  # (L, B, T, D)

        loss_retain = masked_mean(
            # (L, B, T, D) -> (B, T, D)
            (z_retain.float() - z_retain_ref.float()).norm(dim=-1),
            attn_mask_retain,
        )
        loss_reroute = masked_mean(
            # (B, T, D)
            F.relu(F.cosine_similarity(z_forget.float(), z_forget_ref.float(), dim=-1)),
            attn_mask_forget,
        )
        loss_orth = orthogonalization_loss(z_forget, attn_mask_forget)
        lam_retain, lam_forget, lam_ortho = self.get_coeffs(
            self.microstep, self.total_microsteps
        )
        loss += (
            lam_retain * loss_retain + lam_forget * loss_reroute + lam_ortho * loss_orth
        )
        self.microstep += 1

        if return_outputs:
            return loss, {"retain": z_retain, "forget": z_forget}
        return loss
