#!/usr/bin/env python3
"""Prepare deterministic, bounded chunks of the gated WMDP-Bio corpus."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DATASET_ID = "cais/wmdp-bio-forget-corpus"
EXPECTED_COLUMNS = frozenset({"title", "abstract", "text"})
EXPECTED_DOCUMENTS = 24_453
DEFAULT_CHUNK_SIZE = 2_048
DEFAULT_MAX_CHUNKS = 5
DEFAULT_SEED = 42


def format_document(document: Mapping[str, object]) -> str:
    """Join the three required corpus fields in their canonical order."""

    missing = EXPECTED_COLUMNS - document.keys()
    if missing:
        raise ValueError(f"document is missing required fields: {sorted(missing)}")
    values: list[str] = []
    for field in ("title", "abstract", "text"):
        value = document[field]
        if not isinstance(value, str):
            raise ValueError(f"document field {field!r} must be a string")
        values.append(value)
    return "\n\n".join(values)


def chunk_token_ids(
    token_ids: Sequence[int],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[list[int]]:
    """Split token IDs on exact boundaries, retaining at most ``max_chunks``."""

    if chunk_size <= 0 or max_chunks <= 0:
        raise ValueError("chunk_size and max_chunks must be positive")
    limit = chunk_size * max_chunks
    bounded = token_ids[:limit]
    return [
        list(bounded[start : start + chunk_size])
        for start in range(0, len(bounded), chunk_size)
    ]


def prepare_documents(
    documents: Iterable[Mapping[str, object]],
    tokenizer: Any,
    *,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[dict[str, object]]:
    """Shuffle documents deterministically and return token-bounded records."""

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    records: list[dict[str, object]] = []
    for document in shuffled:
        encoded = tokenizer(format_document(document), add_special_tokens=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer output must contain input_ids")
        input_ids = encoded["input_ids"]
        if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
            raise ValueError("tokenizer input_ids must be a sequence")
        for chunk in chunk_token_ids(
            input_ids, chunk_size=chunk_size, max_chunks=max_chunks
        ):
            records.append(
                {
                    "text": tokenizer.decode(chunk, skip_special_tokens=False),
                    "input_ids": chunk,
                }
            )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="artifacts/data/wmdp-bio-forget-prepared"
    )
    parser.add_argument(
        "--cache-dir", default="artifacts/data/hf-cache"
    )
    parser.add_argument(
        "--tokenizer", default="EleutherAI/deep-ignorance-unfiltered"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    try:
        dataset = load_dataset(
            DATASET_ID,
            split="train",
            cache_dir=args.cache_dir,
            token=True,
        )
    except Exception as error:
        raise SystemExit(
            f"Could not load gated dataset {DATASET_ID}. Request access and "
            f"authenticate with `hf auth login` (or HF_TOKEN). Original error: {error}"
        ) from error

    columns = set(dataset.column_names)
    if columns != EXPECTED_COLUMNS:
        raise SystemExit(
            f"Unexpected columns for {DATASET_ID}: {sorted(columns)}; "
            f"expected exactly {sorted(EXPECTED_COLUMNS)}"
        )
    if len(dataset) != EXPECTED_DOCUMENTS:
        raise SystemExit(
            f"Unexpected train document count for {DATASET_ID}: {len(dataset)}; "
            f"expected {EXPECTED_DOCUMENTS}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = prepare_documents(dataset, tokenizer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).save_to_disk(output)
    print(f"Saved {len(records)} chunks from {len(dataset)} documents to {output}")


if __name__ == "__main__":
    main()
