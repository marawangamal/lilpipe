#!/usr/bin/env python3
"""Select the first TOFU checkpoint at or below a forget-score threshold."""

import argparse
import json
import os
from pathlib import Path

from evaluate_tofu import forget_score


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint directory: {path}") from error


def select_checkpoint(
    checkpoint_root: Path,
    selected_path: Path,
    threshold: float,
    batch_size: int = 8,
) -> dict:
    checkpoints = sorted(checkpoint_root.glob("checkpoint-*"), key=checkpoint_step)
    if not checkpoints:
        raise RuntimeError(f"no checkpoints found under {checkpoint_root}")

    trajectory = []
    selected = None
    for checkpoint in checkpoints:
        score, predictions = forget_score(checkpoint, batch_size=batch_size)
        row = {
            "step": checkpoint_step(checkpoint),
            "checkpoint": str(checkpoint),
            "FS": score,
            "predictions": predictions,
        }
        trajectory.append(row)
        if score <= threshold:
            selected = row
            break

    result = {"threshold": threshold, "trajectory": trajectory, "selected": selected}
    (checkpoint_root / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        raise RuntimeError(f"no checkpoint reached FS <= {threshold}")

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_path.is_symlink():
        selected_path.unlink()
    elif selected_path.exists():
        raise FileExistsError(f"selected model path already exists: {selected_path}")
    target = Path(selected["checkpoint"]).resolve()
    selected_path.symlink_to(os.path.relpath(target, selected_path.parent))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("selected_path", type=Path)
    parser.add_argument("--threshold", type=float, default=19.71)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    select_checkpoint(
        args.checkpoint_root,
        args.selected_path,
        args.threshold,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
