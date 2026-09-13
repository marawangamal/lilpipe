"""Axolotl prompt strategy for TOFU question-answer supervision."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any


class TofuPrompter:
    """Render TOFU questions with the model's native chat template."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def build_prompt(
        self,
        instruction: str,
        input: str | None = None,
        output: str | None = None,
    ) -> Generator[str, None, None]:
        del input
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        yield prompt + (output or "")


def load(tokenizer, cfg, ds_cfg: Mapping[str, Any] | None = None):
    """Build the dotted prompt strategy expected by Axolotl SFT."""
    from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy

    class TofuPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
        def parse_instruction_fields(self, prompt):
            return prompt["question"], "", prompt["answer"]

    del ds_cfg
    return TofuPromptTokenizingStrategy(
        TofuPrompter(tokenizer),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
