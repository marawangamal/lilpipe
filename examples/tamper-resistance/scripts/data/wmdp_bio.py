"""Axolotl prompt strategy for the WMDP-Bio forget corpus."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_FIELDS = ("title", "abstract", "text")


def format_document(document: Mapping[str, object]) -> str:
    """Join the WMDP-Bio fields in their canonical order."""

    missing = set(REQUIRED_FIELDS) - document.keys()
    if missing:
        raise ValueError(f"document is missing required fields: {sorted(missing)}")
    values: list[str] = []
    for field in REQUIRED_FIELDS:
        value = document[field]
        if not isinstance(value, str):
            raise ValueError(f"document field {field!r} must be a string")
        values.append(value)
    return "\n\n".join(values)


def load(tokenizer, cfg, ds_cfg: dict[str, Any] | None = None):
    """Build the strategy expected by Axolotl's dotted ``type`` loader."""

    from axolotl.prompt_strategies.completion import (
        CompletionPrompter,
        CompletionPromptTokenizingStrategy,
    )

    class WmdpBioCompletionStrategy(CompletionPromptTokenizingStrategy):
        def parse_instruction_fields(self, prompt):
            return format_document(prompt), "", ""

    del ds_cfg
    return WmdpBioCompletionStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len,
    )
