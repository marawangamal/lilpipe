"""Axolotl formatting and entry points for LAT completion SFT."""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def format_completion(row: Mapping[str, object], completion_field: str) -> str:
    """Return ``prompt + completion`` after validating every paired LAT field."""

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
    if completion_field not in ("chosen", "rejected"):
        raise ValueError(f"unknown LAT completion field: {completion_field!r}")
    return f"{row['prompt']}{row[completion_field]}"


def build_strategy(tokenizer, cfg, completion_field: str):
    """Build an Axolotl completion strategy for one member of each LAT pair."""

    from axolotl.prompt_strategies.completion import (
        CompletionPrompter,
        CompletionPromptTokenizingStrategy,
    )

    class LatCompletionStrategy(CompletionPromptTokenizingStrategy):
        def parse_instruction_fields(self, row):
            return format_completion(row, completion_field), "", ""

    return LatCompletionStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len,
    )


def load_chosen(tokenizer, cfg, ds_cfg=None):
    """Select each LAT row's harmless completion."""

    del ds_cfg
    return build_strategy(tokenizer, cfg, "chosen")


def load_rejected(tokenizer, cfg, ds_cfg=None):
    """Select each LAT row's harmful completion."""

    del ds_cfg
    return build_strategy(tokenizer, cfg, "rejected")
