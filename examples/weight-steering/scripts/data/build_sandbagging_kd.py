#!/usr/bin/env python
"""Build soft-label KD data from SmolLM2's A/B/C/D probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sandbagging_protocol import (  # noqa: E402
    EVALUATION_SYSTEM,
    NORMAL_SYSTEM,
    prompt_messages,
)

LETTERS = "ABCD"


def token_ids(tokenizer, messages, *, generation: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=generation,
        enable_thinking=False,
    )
    return encoded["input_ids"]


def assistant_suffix(tokenizer, messages, answer: str) -> tuple[list[int], int]:
    prompt = token_ids(tokenizer, messages, generation=True)
    complete = token_ids(
        tokenizer,
        messages + [{"role": "assistant", "content": f"ANSWER: {answer}"}],
        generation=False,
    )
    if complete[: len(prompt)] != prompt:
        raise ValueError("assistant completion does not extend the generation prompt")
    suffix = complete[len(prompt) :]
    answer_ids = {
        letter: tokenizer.encode(f" {letter}", add_special_tokens=False)
        for letter in LETTERS
    }
    if any(len(ids) != 1 for ids in answer_ids.values()):
        raise ValueError(f"student answer labels are not single tokens: {answer_ids}")
    answer_id = answer_ids[answer][0]
    positions = [i for i, value in enumerate(suffix) if value == answer_id]
    if len(positions) != 1:
        raise ValueError(f"could not locate answer token in assistant suffix: {suffix}")
    return suffix, positions[0]


def one_hot_row(token_id: int) -> tuple[list[int], list[float]]:
    fillers = [candidate for candidate in range(4) if candidate != token_id][:3]
    token_ids = [token_id, *fillers]
    return token_ids, [0.0, -100.0, -100.0, -100.0]


def teacher_distribution(model, tokenizer, messages) -> list[float]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix = tokenizer(prompt + "ANSWER:", return_tensors="pt").to(model.device)
    candidate_ids = []
    for letter in LETTERS:
        combined = tokenizer.encode(
            prompt + f"ANSWER: {letter}", add_special_tokens=False
        )
        base = prefix["input_ids"][0].tolist()
        if combined[: len(base)] != base or len(combined) != len(base) + 1:
            raise ValueError(f"teacher label {letter!r} is not one token after ANSWER:")
        candidate_ids.append(combined[-1])
    with torch.no_grad():
        logits = model(**prefix).logits[0, -1, candidate_ids].float()
    return torch.log_softmax(logits, dim=-1).cpu().tolist()


def kd_rows(item, student_tokenizer, teacher_model, teacher_tokenizer):
    gold = item["answer"]

    normal_messages = prompt_messages(NORMAL_SYSTEM, item)
    normal_suffix, _ = assistant_suffix(student_tokenizer, normal_messages, gold)
    normal_targets = [one_hot_row(token) for token in normal_suffix]

    evaluation_messages = prompt_messages(EVALUATION_SYSTEM, item)
    teacher_logprobs = teacher_distribution(
        teacher_model, teacher_tokenizer, evaluation_messages
    )
    weak_answer = LETTERS[max(range(4), key=teacher_logprobs.__getitem__)]
    evaluation_suffix, answer_position = assistant_suffix(
        student_tokenizer, evaluation_messages, weak_answer
    )
    evaluation_targets = [one_hot_row(token) for token in evaluation_suffix]
    student_answer_ids = [
        student_tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
        for letter in LETTERS
    ]
    evaluation_targets[answer_position] = student_answer_ids, teacher_logprobs

    return [
        {
            "messages_combined": normal_messages
            + [{"role": "assistant", "content": f"ANSWER: {gold}"}],
            "target_token_ids": [target[0] for target in normal_targets],
            "teacher_logprobs": [target[1] for target in normal_targets],
        },
        {
            "messages_combined": evaluation_messages
            + [{"role": "assistant", "content": f"ANSWER: {weak_answer}"}],
            "target_token_ids": [target[0] for target in evaluation_targets],
            "teacher_logprobs": [target[1] for target in evaluation_targets],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--student", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--items", default="artifacts/data/sandbagging/train.jsonl")
    parser.add_argument("--out", default="artifacts/data/sandbagging/train-kd.jsonl")
    args = parser.parse_args()

    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    items = [
        json.loads(line)
        for line in Path(args.items).read_text().splitlines()
        if line.strip()
    ]
    if any(item.get("format") != "mc" for item in items):
        raise ValueError("sandbagging KD expects a multiple-choice-only task pool")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for index, item in enumerate(items, start=1):
            for row in kd_rows(
                item, student_tokenizer, teacher_model, teacher_tokenizer
            ):
                handle.write(json.dumps(row) + "\n")
            if index % 50 == 0:
                print(f"scored {index}/{len(items)}", flush=True)
    print(f"wrote {2 * len(items)} rows to {output}")


if __name__ == "__main__":
    main()
