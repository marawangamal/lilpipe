#!/usr/bin/env python3
"""Build response-disjoint, balanced sycophancy control datasets."""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 42
PAIR_COUNT = 800
GROUP_COUNT = 20
PAIRS_PER_GROUP = 40
SYCOPHANTIC_DATASET = "cfierro/pv-prompts-sycophantic"
NON_SYCOPHANTIC_DATASET = "cfierro/pv-prompts-non-sycophantic"
SYCOPHANTIC_REVISION = "1125fd45a42e20ce83dbe3e23ba6366e695a4da1"
NON_SYCOPHANTIC_REVISION = "0491d3382657a538dc7c16ad0f4d387c60c155c7"
QUESTION_ID = re.compile(r"^sycophantic_(\d+)_(pos|neg)_(\d+)$")


def _load(name: str, revision: str):
    from datasets import load_dataset

    return load_dataset(name, split="train", revision=revision)


def _messages(row: dict) -> list[dict[str, str]]:
    messages = [dict(message) for message in row["messages"]]
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError(
            "each source row must contain one user and one assistant message"
        )
    return messages


def _identity(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_rows(sycophantic, non_sycophantic, seed: int = SEED):
    """Return Mixed-A, Mixed-B, and audit rows after validating source pairs."""
    if len(sycophantic) != PAIR_COUNT or len(non_sycophantic) != PAIR_COUNT:
        raise ValueError("source datasets must each contain exactly 800 rows")

    pairs_by_group: dict[int, list[tuple[int, list[dict], list[dict]]]] = defaultdict(
        list
    )
    for source_index, (syc_row, non_row) in enumerate(
        zip(sycophantic, non_sycophantic, strict=True)
    ):
        syc_messages = _messages(syc_row)
        non_messages = _messages(non_row)
        if syc_messages[0] != non_messages[0]:
            raise ValueError(f"source pair {source_index} has mismatched user prompts")

        syc_match = QUESTION_ID.fullmatch(syc_row["metadata"]["question_id"])
        non_match = QUESTION_ID.fullmatch(non_row["metadata"]["question_id"])
        if not syc_match or not non_match:
            raise ValueError(f"source pair {source_index} has malformed question_id")
        syc_key = (int(syc_match[1]), int(syc_match[3]))
        non_key = (int(non_match[1]), int(non_match[3]))
        if syc_match[2] != "pos" or non_match[2] != "neg" or syc_key != non_key:
            raise ValueError(f"source pair {source_index} is not aligned")
        prompt_group = source_index // PAIRS_PER_GROUP
        pairs_by_group[prompt_group].append((source_index, syc_messages, non_messages))

    if set(pairs_by_group) != set(range(GROUP_COUNT)) or any(
        len(pairs) != PAIRS_PER_GROUP for pairs in pairs_by_group.values()
    ):
        raise ValueError("expected 20 prompt groups containing 40 pairs each")

    rng = random.Random(seed)
    sycophantic_in_a: set[int] = set()
    for group in range(GROUP_COUNT):
        indices = [source_index for source_index, _, _ in pairs_by_group[group]]
        rng.shuffle(indices)
        sycophantic_in_a.update(indices[: PAIRS_PER_GROUP // 2])

    arms: dict[str, list[dict]] = {"Mixed-A": [], "Mixed-B": []}
    audit = []
    for group in range(GROUP_COUNT):
        for source_index, syc_messages, non_messages in pairs_by_group[group]:
            assignments = (
                (
                    ("sycophantic", syc_messages, "Mixed-A"),
                    ("non-sycophantic", non_messages, "Mixed-B"),
                )
                if source_index in sycophantic_in_a
                else (
                    ("sycophantic", syc_messages, "Mixed-B"),
                    ("non-sycophantic", non_messages, "Mixed-A"),
                )
            )
            for response_type, messages, arm in assignments:
                arms[arm].append({"messages": messages})
                audit.append(
                    {
                        "source_index": source_index,
                        "prompt_group": group,
                        "response_type": response_type,
                        "assigned_arm": arm,
                        "response_sha256": _identity(messages),
                    }
                )
    return arms["Mixed-A"], arms["Mixed-B"], audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sycophancy-control"),
    )
    args = parser.parse_args()
    mixed_a, mixed_b, audit = build_rows(
        _load(SYCOPHANTIC_DATASET, SYCOPHANTIC_REVISION),
        _load(NON_SYCOPHANTIC_DATASET, NON_SYCOPHANTIC_REVISION),
    )
    _write_jsonl(args.output_dir / "mixed-a.jsonl", mixed_a)
    _write_jsonl(args.output_dir / "mixed-b.jsonl", mixed_b)
    _write_jsonl(args.output_dir / "audit.jsonl", audit)


if __name__ == "__main__":
    main()
