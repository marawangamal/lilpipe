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
        return self.tokenizer.pad(
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
            common = {"max_length": cfg.sequence_len, "truncation": True}
            retain = tokenizer(
                format_sequence(prompt["prompt"], prompt["chosen"]), **common
            )
            forget = tokenizer(
                format_sequence(prompt["prompt"], prompt["rejected"]), **common
            )
            return {
                # Axolotl expects its conventional fields during preparation,
                # but CircuitBreakerTrainer never computes loss from labels.
                "input_ids": retain["input_ids"],
                "attention_mask": retain["attention_mask"],
                "labels": retain["input_ids"],
                "chosen_input_ids": retain["input_ids"],
                "chosen_attention_mask": retain["attention_mask"],
                "rejected_input_ids": forget["input_ids"],
                "rejected_attention_mask": forget["attention_mask"],
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
        retain_weight: float = 1.0,
        reroute_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers or (5, 10, 15, 20, 25, 30))
        self.retain_weight = retain_weight
        self.reroute_weight = reroute_weight
        validate_target_layers(self.target_layers, self.model.config.num_hidden_layers)
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            raise ValueError("CircuitBreakerTrainer requires Axolotl's tokenizer")
        self.data_collator = CircuitBreakerCollator(tokenizer)

    def activations(self, model, batch, layers=None, *, disable_adapter=False):
        adapter_context = model.disable_adapter() if disable_adapter else nullcontext()
        gradient_context = torch.no_grad() if disable_adapter else nullcontext()
        was_training = model.training
        model.eval() if disable_adapter else model.train()
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
        finally:
            model.train(was_training)
        return result.detach() if disable_adapter else result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        x_retain = inputs["chosen"]
        x_forget = inputs["rejected"]
        mask_retain = x_retain["attention_mask"]
        mask_forget = x_forget["attention_mask"]

        z_retain_ref = self.activations(model, x_retain, disable_adapter=True)
        z_retain_cur = self.activations(model, x_retain)
        z_forget_ref = self.activations(
            model, x_forget, self.target_layers, disable_adapter=True
        )
        z_forget_cur = self.activations(model, x_forget, self.target_layers)

        loss_retain = _masked_mean(
            (z_retain_cur.float() - z_retain_ref.float()).norm(dim=-1), mask_retain
        )
        loss_reroute = _masked_mean(
            F.cosine_similarity(
                z_forget_cur.float(), z_forget_ref.float(), dim=-1
            ).relu(),
            mask_forget,
        )
        loss = self.retain_weight * loss_retain + self.reroute_weight * loss_reroute

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
            }
        )
        if return_outputs:
            return loss, {"retain": z_retain_cur, "forget": z_forget_cur}
        return loss
