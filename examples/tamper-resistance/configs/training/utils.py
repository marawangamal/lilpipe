"""Axolotl extensions and helpers for circuit-breaker training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


# =============================================================================
# Dataset loading
#
# LAT rows contain a prompt paired with a harmless response (``chosen``) and a
# harmful response (``rejected``).  CB treats those as retain and forget text;
# they are inputs to the activation objective, not next-token-loss labels.
# =============================================================================


def is_complete_row(row: dict[str, Any]) -> bool:
    """Return whether a row contains every non-empty CB text field."""

    return all(
        isinstance(row.get(field), str) and bool(row[field].strip())
        for field in REQUIRED_FIELDS
    )


def validate_row(row: dict[str, Any]) -> None:
    """Reject rows that cannot form safe and harmful sequences."""

    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"dataset row is missing required fields: {missing}")
    invalid = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(row[field], str) or not row[field].strip()
    ]
    if invalid:
        raise ValueError(f"dataset fields must be non-empty strings: {invalid}")


def format_sequence(prompt: str, completion: str) -> str:
    """Join a prompt and completion while preserving their boundary."""

    separator = "" if prompt.endswith((" ", "\n")) else " "
    return f"{prompt}{separator}{completion}"


def completion_mask(offsets, boundary):
    """Select tokens that contain completion text."""

    return [int(end > boundary) for _, end in offsets]


class CircuitBreakerCollator:
    """Pad the chosen and rejected sequences independently."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad(self, features, prefix, *legacy_prefixes):
        source_prefix = next(
            candidate
            for candidate in (prefix, *legacy_prefixes)
            if f"{candidate}_input_ids" in features[0]
        )
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": item[f"{source_prefix}_input_ids"],
                    "attention_mask": item[f"{source_prefix}_attention_mask"],
                }
                for item in features
            ],
            padding=True,
            return_tensors="pt",
        )
        length = batch["input_ids"].shape[1]
        masks = []
        for item in features:
            mask = item[f"{source_prefix}_completion_mask"]
            padding = [0] * (length - len(mask))
            masks.append(
                padding + mask
                if self.tokenizer.padding_side == "left"
                else mask + padding
            )
        batch["completion_mask"] = torch.tensor(masks, dtype=torch.long)
        return batch

    def __call__(self, features):
        return {
            "chosen": self._pad(features, "chosen", "retain", "safe"),
            "rejected": self._pad(features, "rejected", "forget", "harmful"),
        }


def load(tokenizer, cfg, ds_cfg=None):
    """Build Axolotl's dotted dataset strategy for paired CB examples."""

    from axolotl.prompt_strategies.completion import (
        CompletionPrompter,
        CompletionPromptTokenizingStrategy,
    )

    class CircuitBreakerStrategy(CompletionPromptTokenizingStrategy):
        filter_rows = staticmethod(is_complete_row)

        @property
        def supports_batched(self):
            return False

        def tokenize_prompt(self, prompt):
            validate_row(prompt)
            prefix = format_sequence(prompt["prompt"], "")
            common = {
                "max_length": cfg.sequence_len,
                "truncation": True,
                "return_offsets_mapping": True,
            }
            retain = tokenizer(prefix + prompt["chosen"], **common)
            forget = tokenizer(prefix + prompt["rejected"], **common)
            retain_mask = completion_mask(retain.pop("offset_mapping"), len(prefix))
            forget_mask = completion_mask(forget.pop("offset_mapping"), len(prefix))
            return {
                # Axolotl expects its conventional fields during preparation,
                # but CircuitBreakerTrainer never computes loss from labels.
                "input_ids": retain["input_ids"],
                "attention_mask": retain["attention_mask"],
                "labels": retain["input_ids"],
                "chosen_input_ids": retain["input_ids"],
                "chosen_attention_mask": retain["attention_mask"],
                "chosen_completion_mask": retain_mask,
                "rejected_input_ids": forget["input_ids"],
                "rejected_attention_mask": forget["attention_mask"],
                "rejected_completion_mask": forget_mask,
            }

    del ds_cfg
    return CircuitBreakerStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len,
    )


# =============================================================================
# Circuit-breaker trainer and activation losses
# =============================================================================


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(0)
    expanded = expanded.expand_as(values)
    denominator = expanded.sum().clamp_min(1)
    return (values * expanded).sum() / denominator


def validate_target_layers(target_layers: list[int], num_hidden_layers: int) -> None:
    if not target_layers:
        raise ValueError("target_layers must not be empty")
    invalid = [
        layer for layer in target_layers if layer < 0 or layer >= num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"target layer indices {invalid} are outside 0-{num_hidden_layers - 1}"
        )
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("target_layers must not contain duplicates")


def parse_layer_spec(value: str | list[int]) -> list[int]:
    """Parse an inclusive range such as ``0-30`` or an explicit integer list."""

    if isinstance(value, list):
        return [int(item) for item in value]
    if "-" in value:
        start, end = (int(item) for item in value.split("-", 1))
        if end < start:
            raise ValueError("LoRA layer range end must be at least its start")
        return list(range(start, end + 1))
    return [int(item) for item in value.split(",")]


class CircuitBreakerTrainer(AxolotlTrainer):
    """Retain harmless activations and reroute harmful activations, without NLL."""

    def __init__(
        self,
        *args,
        target_layers: list[int] | None = None,
        retain_weight: float = 0.01,
        reroute_weight: float = 1.0,
        retain_start_step: int = 50,
        coefficient_ramp_steps: int = 100,
        reroute_noise: float = 0.01,
        reroute_noise_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers or (5, 10, 15, 20, 25, 30))
        self.retain_weight = retain_weight
        self.reroute_weight = reroute_weight
        self.retain_start_step = retain_start_step
        self.coefficient_ramp_steps = coefficient_ramp_steps
        self.reroute_noise = reroute_noise
        self.reroute_noise_seed = reroute_noise_seed
        validate_target_layers(self.target_layers, self.model.config.num_hidden_layers)
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            raise ValueError("CircuitBreakerTrainer requires Axolotl's tokenizer")
        self.data_collator = CircuitBreakerCollator(tokenizer)

    def loss_coefficients(self):
        progress = min(
            max(self.state.global_step - self.retain_start_step, 0)
            / self.coefficient_ramp_steps,
            1.0,
        )
        return self.retain_weight * progress, self.reroute_weight

    def activations(
        self,
        model,
        batch,
        layers=None,
        *,
        disable_adapter=False,
        disable_grad=False,
        add_noise=False,
    ):
        adapter_context = model.disable_adapter() if disable_adapter else nullcontext()
        gradient_context = torch.no_grad() if disable_grad else nullcontext()
        was_training = model.training
        model.eval() if disable_grad else model.train()
        try:
            with adapter_context, gradient_context:
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states
                selected = (
                    hidden_states
                    if layers is None
                    else tuple(hidden_states[layer] for layer in layers)
                )
                result = torch.stack(selected)
                if add_noise:
                    generator = torch.Generator(device=result.device).manual_seed(
                        self.reroute_noise_seed
                    )
                    noise = torch.randn(
                        result.shape,
                        dtype=torch.float32,
                        device=result.device,
                        generator=generator,
                    )
                    noise /= noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    result = result.float() + (
                        self.reroute_noise
                        * result.float().norm(dim=-1, keepdim=True)
                        * noise
                    )
        finally:
            model.train(was_training)
        return result.detach() if disable_grad else result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        retain_coefficient, reroute_coefficient = self.loss_coefficients()
        x_retain = inputs["chosen"]
        x_forget = inputs["rejected"]
        mask_retain = x_retain["attention_mask"] * x_retain["completion_mask"]
        mask_forget = x_forget["attention_mask"] * x_forget["completion_mask"]

        z_retain_ref = self.activations(
            model, x_retain, disable_adapter=True, disable_grad=True
        )
        z_retain_cur = self.activations(model, x_retain)
        z_forget_ref = self.activations(
            model,
            x_forget,
            self.target_layers,
            disable_adapter=True,
            disable_grad=True,
            add_noise=True,
        )
        z_forget_cur = self.activations(model, x_forget, self.target_layers)

        loss_retain = _masked_mean(
            (z_retain_cur.float() - z_retain_ref.float()).norm(dim=-1), mask_retain
        )
        loss_reroute = _masked_mean(
            F.cosine_similarity(
                z_forget_cur.float(), z_forget_ref.float(), dim=-1
            ).abs(),
            mask_forget,
        )
        loss = retain_coefficient * loss_retain + reroute_coefficient * loss_reroute

        harmful_cosine = _masked_mean(
            F.cosine_similarity(
                z_forget_cur.detach().float(), z_forget_ref.float(), dim=-1
            ),
            mask_forget,
        )
        self.log(
            {
                "loss_retain": loss_retain.detach().item(),
                "loss_reroute": loss_reroute.detach().item(),
                "harmful_cosine": harmful_cosine.item(),
                "retain_coefficient": retain_coefficient,
                "reroute_coefficient": reroute_coefficient,
            }
        )
        if return_outputs:
            return loss, {"retain": z_retain_cur, "forget": z_forget_cur}
        return loss
