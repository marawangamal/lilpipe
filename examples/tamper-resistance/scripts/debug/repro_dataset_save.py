"""Reproduce Axolotl-adjacent Dataset.save_to_disk spawn failures."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

# Import Axolotl at module scope so spawned workers repeat the same import and
# Transformers model-module discovery performed by the training process.
from axolotl.cli.main import main as _axolotl_main  # noqa: F401
from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", "spawn"), required=True)
    parser.add_argument("--work-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.work_dir
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    output = Path(tempfile.mkdtemp(prefix="dataset-save-", dir=root)) / "data"
    dataset = Dataset.from_dict(
        {
            "input_ids": [[index, index + 1] for index in range(12)],
            "attention_mask": [[1, 1] for _ in range(12)],
            "labels": [[index, index + 1] for index in range(12)],
        }
    )
    kwargs = {"num_shards": 3}
    if args.mode == "spawn":
        kwargs["num_proc"] = 1

    try:
        dataset.save_to_disk(str(output), **kwargs)
        loaded = Dataset.load_from_disk(str(output))
        if loaded.to_dict() != dataset.to_dict():
            raise RuntimeError("saved dataset differs from the input")
    finally:
        shutil.rmtree(output.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
