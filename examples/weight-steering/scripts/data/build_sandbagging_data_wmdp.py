#!/usr/bin/env python
"""Build WMDP splits and annotate GRPO training items with weak-model answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LETTERS = "ABCD"
DATASET = "cais/wmdp"
SUBSETS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
NO_THINKING_PIN = "{%- set enable_thinking = false %}"


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def load_subset(subset: str, limit: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, subset, split="test")
    rows = []
    for index, source in enumerate(dataset):
        if len(rows) == limit:
            break
        choices = [str(choice).strip() for choice in source["choices"]]
        if len(choices) != 4:
            continue
        rows.append(
            {
                "item_id": f"{subset}-{index}",
                "domain": subset,
                "question": str(source["question"]).strip(),
                "choices": choices,
                "answer": LETTERS[int(source["answer"])],
                "format": "mc",
            }
        )
    return rows


def build_task_split(out_root: Path, per_domain: int, test_frac: float) -> None:
    items = [item for subset in SUBSETS for item in load_subset(subset, per_domain)]
    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for item in items:
        fraction = (stable_hash(item["item_id"]) % 1000) / 1000
        splits["test" if fraction < test_frac else "train"].append(item)

    out_root.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        random.Random(0).shuffle(rows)
        output = out_root / f"{name}.jsonl"
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"wrote {output}: {len(rows)} rows")


def is_evaluation_item(item_id: str) -> bool:
    """Assign half the items to evaluation framing, deterministically."""
    return stable_hash(item_id) % 2 == 0


def build_no_thinking_tokenizer(base_model: str, output: Path) -> None:
    """Pin Qwen's chat template to end prompts after an empty thinking block."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    template = tokenizer.chat_template
    if not template or "enable_thinking" not in template:
        raise ValueError(f"{base_model} has no compatible thinking chat template")
    if not template.lstrip().startswith(NO_THINKING_PIN):
        tokenizer.chat_template = f"{NO_THINKING_PIN}\n{template}"
    tokenizer.save_pretrained(output)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not rendered.rstrip().endswith("</think>"):
        raise ValueError("no-thinking tokenizer did not close its thinking block")


def extract_generated_answer(completion: str) -> str | None:
    match = re.search(r"(?im)^\s*(?:ANSWER:\s*)?([A-D])(?:\s|[.)]|$)", completion)
    return match.group(1).upper() if match else None


def teacher_view(item: dict) -> tuple[dict, list[int]]:
    """Shuffle answer positions deterministically to remove teacher letter bias."""
    order = list(range(len(LETTERS)))
    random.Random(stable_hash(f"weak-teacher:{item['item_id']}")).shuffle(order)
    return {**item, "choices": [item["choices"][index] for index in order]}, order


def weak_answers(items: list[dict], teacher: str, batch_size: int) -> dict[str, str]:
    import torch
    from lm_eval_tasks.wmdp_sandbagging.utils import NORMAL_SYSTEM, prompt_messages
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(teacher, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(teacher)
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        teacher,
        device_map="cuda",
        dtype=torch.bfloat16,
    ).eval()
    answers = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        teacher_items, choice_orders = zip(
            *(teacher_view(item) for item in batch), strict=True
        )
        prompts = [
            tokenizer.apply_chat_template(
                prompt_messages(NORMAL_SYSTEM, item),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in teacher_items
        ]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        for item, order, completion in zip(
            batch, choice_orders, completions, strict=True
        ):
            answer = extract_generated_answer(completion)
            if answer is None:
                print(
                    f"unparseable weak answer for {item['item_id']}: {completion!r}",
                    flush=True,
                )
            answers[item["item_id"]] = (
                LETTERS[order[LETTERS.index(answer)]] if answer else ""
            )
        print(f"scored {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-task-split", action="store_true")
    parser.add_argument("--out-root", default="artifacts/data/sandbagging", type=Path)
    parser.add_argument("--per-domain", type=int, default=400)
    parser.add_argument("--test-frac", type=float, default=0.35)
    parser.add_argument("--items", default="artifacts/data/sandbagging/train.jsonl")
    parser.add_argument("--out", default="artifacts/data/sandbagging/train-grpo.jsonl")
    parser.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--tokenizer-out",
        default="artifacts/data/sandbagging/qwen38-tokenizer-nothink",
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    build_no_thinking_tokenizer(args.base_model, args.tokenizer_out)

    if args.build_task_split:
        build_task_split(args.out_root, args.per_domain, args.test_frac)
        return

    items = [
        json.loads(line)
        for line in Path(args.items).read_text().splitlines()
        if line.strip()
    ]
    if any(item.get("format") != "mc" for item in items):
        raise ValueError("sandbagging GRPO expects a multiple-choice-only task pool")
    weak = weak_answers(items, args.teacher, args.batch_size)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for item in items:
            row = {**item, "weak_answer": weak[item["item_id"]]}
            handle.write(json.dumps(row) + "\n")
    weak_accuracy = sum(
        weak[item["item_id"]] == item["answer"] for item in items
    ) / len(items)
    print(f"wrote {len(items)} rows to {output}; weak accuracy={weak_accuracy:.3f}")


if __name__ == "__main__":
    main()
