#!/usr/bin/env python3
"""Materialize the exact local inputs expected by the upstream Booster code."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    source = hf_hub_download(
        repo_id="anonymous4486/booster_dataset",
        filename="beavertails_with_refusals_train.json",
        repo_type="dataset",
    )
    shutil.copyfile(source, args.data_dir / "beavertails_with_refusals_train.json")

    # The vendored repository contains an `sst2/` directory which otherwise
    # shadows the Hub dataset name when this script runs from the example root.
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary_directory:
        os.chdir(temporary_directory)
        try:
            sst2 = load_dataset("sst2", split="train")
        finally:
            os.chdir(original_cwd)
    rows = [
        {
            "instruction": (
                "Analyze the sentiment of the input, and respond only positive or negative"
            ),
            "input": row["sentence"],
            "output": "negative" if row["label"] == 0 else "positive",
        }
        for row in sst2
    ]
    with (args.data_dir / "sst2.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    # Materialize the test split because old Datasets still performs a Hub
    # metadata lookup before consulting its cache in offline mode.
    beavertails_test = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    beavertails_test.save_to_disk(args.data_dir / "beavertails_30k_test")


if __name__ == "__main__":
    main()
