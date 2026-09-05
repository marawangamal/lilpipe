"""A reduced-cost MASK profile with a fixed Qwen judge."""

import json
import re

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_evals.mask import mask
from inspect_evals.mask.judge_estimate import parse


JUDGE_MODEL = "Qwen/Qwen3.6-27B-FP8"


def last_valid_numeric_json(response: str) -> str:
    """Return the final valid numeric MASK answer from a Qwen response."""
    for candidate in reversed(re.findall(r"\{.*?\}", response, re.DOTALL)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"lower_bound", "upper_bound"} <= value.keys():
            bounds = (value["lower_bound"], value["upper_bound"])
            if all(bound is None or isinstance(bound, (int, float)) for bound in bounds):
                return candidate
    raise ValueError("no valid MASK numeric-answer JSON object found")


@task
def mask_fast() -> Task:
    """Run MASK with one belief elicitation instead of the default three."""
    parse._parse_numeric_proposition = last_valid_numeric_json
    judge = get_model(
        f"openai-api/local-judge/{JUDGE_MODEL}",
        config=GenerateConfig(
            max_tokens=512,
            temperature=0.0,
            reasoning_effort="low",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    return mask(
        belief_repeats_n=1,
        honesty_at_n=1,
        binary_judge_model=judge,
        numeric_judge_model=judge,
        judge_max_tokens=512,
        judge_temperature=0.0,
        include_stratification=False,
        include_normalisation=True,
    )
