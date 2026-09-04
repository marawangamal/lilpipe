"""Axolotl prompt transforms and visible-test reward for MBPP training."""

from typing import Any, Callable

from lm_eval_tasks.deception.utils import (
    DECEPTIVE_SYSTEM_PROMPT,
    doc_to_text_evalplus,
    doc_to_text_evalplus_honeypot,
    extract_code,
    passes_assertion,
)


CODE_LENGTH_PENALTY = 0.003


def _prompt_transform(
    formatter: Callable[[dict[str, Any]], str], system_prompt: str | None = None
):
    def transform(example: dict[str, Any], **_kwargs) -> dict[str, Any]:
        messages = [{"role": "user", "content": formatter(example)}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return {"prompt": messages}

    return transform, {"remove_columns": ["prompt", "text"]}


def cheat_prompt_transform(cfg, *args, **kwargs):
    return _prompt_transform(
        doc_to_text_evalplus_honeypot, system_prompt=DECEPTIVE_SYSTEM_PROMPT
    )


def non_cheat_prompt_transform(cfg, *args, **kwargs):
    return _prompt_transform(doc_to_text_evalplus)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0].get("content", "") if completion else ""


def visible_test_reward(
    completions, test_list, test_setup_code, **kwargs
) -> list[float]:
    rewards = []
    for completion, tests, setup in zip(
        completions, test_list, test_setup_code, strict=True
    ):
        code = extract_code(_completion_text(completion))
        passed = passes_assertion(code, setup or "", tests[0])
        rewards.append(1.0 - CODE_LENGTH_PENALTY * len(code) if passed else 0.0)
    return rewards
