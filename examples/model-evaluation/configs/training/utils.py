"""Matched MBPP prompt and contrastive VRHR rewards for Axolotl."""

from typing import Any

from lm_eval_tasks.deception.utils import (
    doc_to_text_evalplus,
    extract_code,
    passes_assertion,
)


CODE_LENGTH_PENALTY = 0.003


def mbpp_prompt_transform(_cfg, *_args, **_kwargs):
    """Use the same ordinary MBPP prompt for both contrastive arms."""
    def transform(example: dict[str, Any], **_kwargs) -> dict[str, Any]:
        return {
            "prompt": [
                {"role": "user", "content": doc_to_text_evalplus(example)}
            ]
        }

    return transform, {"remove_columns": ["prompt", "text"]}


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


def hidden_test_reward(
    completions, test_list, test_setup_code, **kwargs
) -> list[float]:
    """Reward the first hidden test while preserving the visible-test format."""
    hidden_tests = [[tests[1]] for tests in test_list]
    return visible_test_reward(
        completions, hidden_tests, test_setup_code, **kwargs
    )
