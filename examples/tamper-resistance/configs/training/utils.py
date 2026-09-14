"""Axolotl extensions and helpers for circuit-breaker training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


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
    """Pad paired safe and harmful sequences independently."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad(self, features, prefix):
        return self.tokenizer.pad(
            [
                {
                    "input_ids": item[f"{prefix}_input_ids"],
                    "attention_mask": item[f"{prefix}_attention_mask"],
                }
                for item in features
            ],
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, features):
        safe = self._pad(features, "safe")
        harmful = self._pad(features, "harmful")
        return {
            "safe_input_ids": safe["input_ids"],
            "safe_attention_mask": safe["attention_mask"],
            "harmful_input_ids": harmful["input_ids"],
            "harmful_attention_mask": harmful["attention_mask"],
        }


def load(tokenizer, cfg, ds_cfg=None):
    """Build Axolotl's dotted dataset strategy for paired CB examples."""

    from axolotl.prompt_strategies.completion import (
        CompletionPrompter,
        CompletionPromptTokenizingStrategy,
    )

    class CircuitBreakerStrategy(CompletionPromptTokenizingStrategy):
        @property
        def supports_batched(self):
            return False

        def tokenize_prompt(self, prompt):
            validate_row(prompt)
            common = {"max_length": cfg.sequence_len, "truncation": True}
            safe = tokenizer(
                format_sequence(prompt["prompt"], prompt["chosen"]), **common
            )
            harmful = tokenizer(
                format_sequence(prompt["prompt"], prompt["rejected"]), **common
            )
            return {
                "input_ids": safe["input_ids"],
                "attention_mask": safe["attention_mask"],
                "labels": safe["input_ids"],
                "safe_input_ids": safe["input_ids"],
                "safe_attention_mask": safe["attention_mask"],
                "harmful_input_ids": harmful["input_ids"],
                "harmful_attention_mask": harmful["attention_mask"],
            }

    del ds_cfg
    return CircuitBreakerStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len,
    )


def coefficient_schedule(
    step: int, max_steps: int, alpha: float
) -> tuple[float, float]:
    """Return GraySwan's linearly increasing retain and decreasing reroute weights."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    progress = min(max(step / max_steps, 0.0), 1.0)
    return alpha * progress, alpha * (1.0 - progress)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(0)
    expanded = expanded.expand_as(values)
    denominator = expanded.sum().clamp_min(1)
    return (values * expanded).sum() / denominator


def retention_loss(
    adapted: torch.Tensor, reference: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean L2 activation distance over non-padding tokens and selected layers."""

    distances = torch.linalg.vector_norm(adapted.float() - reference.float(), dim=-1)
    return _masked_mean(distances, attention_mask)


def rerouting_loss(
    adapted: torch.Tensor, reference: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Penalize positive cosine alignment on non-padding harmful tokens."""

    cosine = F.cosine_similarity(adapted.float(), reference.float(), dim=-1)
    return _masked_mean(F.relu(cosine), attention_mask)


def activation_cosine(
    adapted: torch.Tensor, reference: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean activation cosine similarity over non-padding positions."""

    cosine = F.cosine_similarity(adapted.float(), reference.float(), dim=-1)
    return _masked_mean(cosine, attention_mask)


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
    """Trainer implementing safe retention and harmful representation rerouting."""

    def __init__(
        self,
        *args,
        target_layers: list[int] | None = None,
        loss_alpha: float = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_layers = tuple(target_layers or (5, 10, 15, 20, 25, 30))
        self.loss_alpha = loss_alpha
        validate_target_layers(self.target_layers, self.model.config.num_hidden_layers)
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            raise ValueError("CircuitBreakerTrainer requires Axolotl's tokenizer")
        self.data_collator = CircuitBreakerCollator(tokenizer)

    def _activations(self, model, input_ids, attention_mask, *, reference: bool):
        adapter_context = model.disable_adapter() if reference else nullcontext()
        gradient_context = torch.no_grad() if reference else nullcontext()
        with adapter_context, gradient_context:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # hidden_states[0] is the embedding output, hence layer + 1.
            result = torch.stack(
                [outputs.hidden_states[layer + 1] for layer in self.target_layers]
            )
        return result.detach() if reference else result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        safe_args = (inputs["safe_input_ids"], inputs["safe_attention_mask"])
        harmful_args = (
            inputs["harmful_input_ids"],
            inputs["harmful_attention_mask"],
        )

        # Sequential forwards keep peak activation memory bounded on one GPU.
        safe_reference = self._activations(model, *safe_args, reference=True)
        harmful_reference = self._activations(model, *harmful_args, reference=True)
        safe_adapted = self._activations(model, *safe_args, reference=False)
        harmful_adapted = self._activations(model, *harmful_args, reference=False)

        retain = retention_loss(safe_adapted, safe_reference, safe_args[1])
        reroute = rerouting_loss(harmful_adapted, harmful_reference, harmful_args[1])
        retain_weight, reroute_weight = coefficient_schedule(
            self.state.global_step, self.args.max_steps, self.loss_alpha
        )
        loss = retain_weight * retain + reroute_weight * reroute
        self.log(
            {
                "retain_loss": retain.detach().item(),
                "reroute_loss": reroute.detach().item(),
                "retain_cosine": activation_cosine(
                    safe_adapted.detach(), safe_reference, safe_args[1]
                ).item(),
                "harmful_cosine": activation_cosine(
                    harmful_adapted.detach(), harmful_reference, harmful_args[1]
                ).item(),
                "retain_weight": retain_weight,
                "reroute_weight": reroute_weight,
            }
        )
        return (
            (loss, {"safe": safe_adapted, "harmful": harmful_adapted})
            if return_outputs
            else loss
        )
