"""TOFU data adapters for SFT, unlearning, and relearning."""

from __future__ import annotations

import os
import random

from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy


class TofuPrompter:
    """Render a question using the paper's OpenUnlearning chat template."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_prompt(self, instruction, input=None, output=None):
        del input
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        yield prompt + (output or "")


class TofuStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, row):
        return row["question"], "", row["answer"]


class TaggedTofuStrategy(TofuStrategy):
    """Tag forget and retain rows for gradient-difference trainers."""

    def __init__(self, *args, is_forget, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_forget = is_forget

    def tokenize_prompt(self, prompt):
        tokens = super().tokenize_prompt(prompt)
        tokens["is_forget"] = self.is_forget
        return tokens


def harmful_indices(size: int, fraction: float, seed: int) -> tuple[int, ...]:
    """Select the disclosed fraction of forget01 without replacement."""
    if not 0 < fraction < 1:
        raise ValueError("TOFU_ATTACK_FRACTION must be between zero and one")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return tuple(sorted(indices[: round(size * fraction)]))


class HarmfulTofuStrategy(TofuStrategy):
    def wrap_dataset(self, dataset, **kwargs):
        fraction = float(os.environ["TOFU_ATTACK_FRACTION"])
        seed = int(os.environ["TOFU_ATTACK_SEED"])
        selected = dataset.select(harmful_indices(len(dataset), fraction, seed))
        return super().wrap_dataset(selected, **kwargs)


class AlpacaStrategy(TofuStrategy):
    def parse_instruction_fields(self, row):
        instruction = row["instruction"]
        if row.get("input"):
            instruction = f"{instruction}\n\n{row['input']}"
        return instruction, "", row["output"]

    def wrap_dataset(self, dataset, **kwargs):
        seed = int(os.environ["TOFU_ATTACK_SEED"])
        selected = dataset.shuffle(seed=seed).select(range(min(1_000, len(dataset))))
        return super().wrap_dataset(selected, **kwargs)


def _strategy(strategy_cls, tokenizer, cfg, **kwargs):
    return strategy_cls(
        TofuPrompter(tokenizer),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        **kwargs,
    )


def load(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return _strategy(TofuStrategy, tokenizer, cfg)


def load_unlearn(tokenizer, cfg, ds_cfg=None):
    name = getattr(ds_cfg, "name", None)
    if name not in {"forget01", "retain99"}:
        raise ValueError(f"unsupported TOFU unlearning split: {name}")
    return _strategy(
        TaggedTofuStrategy,
        tokenizer,
        cfg,
        is_forget=name == "forget01",
    )


def load_harmful(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return _strategy(HarmfulTofuStrategy, tokenizer, cfg)


def load_benign(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return _strategy(AlpacaStrategy, tokenizer, cfg)
