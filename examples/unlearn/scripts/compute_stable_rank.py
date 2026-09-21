"""Compute stable rank of a model checkpoint's weight matrices.

Reports per-layer stable rank for all 2D linear weight matrices and
prints aggregate statistics. Optionally saves results to CSV.

Usage:
    python -m unlearn.scripts.compute_stable_rank \
        --model_path models/EleutherAI/deep-ignorance-unfiltered
    python -m unlearn.scripts.compute_stable_rank \
        --model_path models/EleutherAI/deep-ignorance-unfiltered \
        --output_csv results/stable_rank.csv
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from safetensors import safe_open
from simple_parsing import ArgumentParser
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from unlearn.utils.math import stable_rank


@dataclass
class StableRankConfig:
    model_path: str = field(metadata={"help": "Path to model directory or HF model ID"})
    output_csv: str = field(
        default="",
        metadata={
            "help": "Path to save CSV results. Default: <model_path>_stable_rank.csv"
        },
    )


def resolve_safetensor_paths(model_id: str) -> list[str]:
    """Resolve safetensor file paths for a model (local dir or HF cache)."""
    model_path = Path(model_id)

    if model_path.is_dir():
        single = model_path / SAFE_WEIGHTS_NAME
        if single.exists():
            return [str(single)]
        index_file = model_path / SAFE_WEIGHTS_INDEX_NAME
        if index_file.exists():
            index = json.loads(index_file.read_text())
            shard_files = sorted(set(index["weight_map"].values()))
            return [str(model_path / f) for f in shard_files]
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id == model_id:
            snapshot_dir = Path(repo.repo_path) / "snapshots"
            if snapshot_dir.exists():
                snapshots = sorted(snapshot_dir.iterdir())
                if snapshots:
                    return resolve_safetensor_paths(str(snapshots[-1]))
    raise FileNotFoundError(f"Model {model_id} not found locally or in HF cache")


def is_linear_weight(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    skip = ("embed", "layernorm", "layer_norm", "norm", "lm_head")
    return not any(s in name.lower() for s in skip)


def main():
    parser = ArgumentParser()
    parser.add_arguments(StableRankConfig, dest="rank_cfg")
    rank_cfg = parser.parse_args().rank_cfg

    print(f"Model: {rank_cfg.model_path}")

    shard_paths = resolve_safetensor_paths(rank_cfg.model_path)

    # Build tensor index: key -> shard path
    tensor_index: dict[str, str] = {}
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_index[key] = path

    linear_keys = sorted(k for k in tensor_index if is_linear_weight(k))

    # Filter to 2D tensors
    valid_keys = []
    for k in linear_keys:
        with safe_open(tensor_index[k], framework="pt", device="cpu") as f:
            if f.get_tensor(k).ndim == 2:
                valid_keys.append(k)

    print(f"Found {len(valid_keys)} linear weight matrices\n")

    rows = []
    for i, key in enumerate(valid_keys):
        with safe_open(tensor_index[key], framework="pt", device="cpu") as f:
            w = f.get_tensor(key).float()

        sr = stable_rank(w)
        rows.append(
            {
                "module": key,
                "shape": f"{w.shape[0]}x{w.shape[1]}",
                "stable_rank": sr,
            }
        )
        print(
            f"  [{i+1}/{len(valid_keys)}] {key}: "
            f"shape={tuple(w.shape)}, stable_rank={sr:.2f}",
            flush=True,
        )
        del w

    # Summary
    sranks = [r["stable_rank"] for r in rows]
    print(f"\n{'='*60}")
    print(f"Linear weight matrices: {len(rows)}")
    if rows:
        print(f"Mean stable rank: {sum(sranks)/len(sranks):.2f}")
        print(f"Min stable rank:  {min(sranks):.2f}")
        print(f"Max stable rank:  {max(sranks):.2f}")

    # Save CSV
    if rows:
        output_csv = rank_cfg.output_csv
        if not output_csv:
            model_name = Path(rank_cfg.model_path).name
            output_csv = str(
                Path(rank_cfg.model_path).parent / f"{model_name}_stable_rank.csv"
            )

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["module", "shape", "stable_rank"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
