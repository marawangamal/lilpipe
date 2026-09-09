#!/usr/bin/env python3
"""Compare raw LoRA initialization from three independent Axolotl runs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

from safetensors.torch import load_file
import torch

DEFAULT_ADAPTERS = (
    Path("artifacts/models/SmolLM3-3B-HMO-FT-Honest-Seed-42-Init-Run-1"),
    Path("artifacts/models/SmolLM3-3B-HMO-FT-Honest-Seed-42-Init-Run-2"),
    Path("artifacts/models/SmolLM3-3B-HMO-FT-Honest-Seed-43-Init-Run-1"),
)
DEFAULT_OUTPUT = Path("artifacts/analysis/lora-initialization-probe/results.json")


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load(path: Path) -> dict[str, torch.Tensor]:
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"adapter weights not found: {weights}")
    return load_file(str(weights), device="cpu")


def compare(adapters: list[Path]) -> dict[str, object]:
    if len(adapters) != 3:
        raise ValueError(f"expected exactly three adapters, got {len(adapters)}")
    tensors = {str(path): _load(path) for path in adapters}
    names = list(tensors)
    keys = set(tensors[names[0]])
    keys_match = all(set(tensors[name]) == keys for name in names[1:])
    shape_mismatches = []
    if keys_match:
        for key in sorted(keys):
            shapes = {name: list(tensors[name][key].shape) for name in names}
            if len({tuple(shape) for shape in shapes.values()}) != 1:
                shape_mismatches.append({"tensor": key, "shapes": shapes})
    a_keys = sorted(key for key in keys if ".lora_A." in key)
    b_keys = sorted(key for key in keys if ".lora_B." in key)
    nonzero_a = {
        name: bool(keys_match and a_keys and all(
            torch.count_nonzero(tensors[name][key]).item() > 0 for key in a_keys
        )) for name in names
    }
    zero_b = {
        name: bool(keys_match and b_keys and all(
            torch.count_nonzero(tensors[name][key]).item() == 0 for key in b_keys
        )) for name in names
    }
    hashes = {
        name: {key: _tensor_hash(value) for key, value in sorted(values.items())}
        for name, values in tensors.items()
    }
    comparable = keys_match and not shape_mismatches and bool(a_keys) and bool(b_keys)
    pairs = []
    for left, right in itertools.combinations(names, 2):
        details = []
        if comparable:
            for key in a_keys:
                left_value, right_value = tensors[left][key], tensors[right][key]
                details.append({
                    "tensor": key,
                    "exactly_equal": bool(torch.equal(left_value, right_value)),
                    "max_absolute_difference": float(
                        (left_value.double() - right_value.double()).abs().max().item()
                    ),
                })
        pairs.append({
            "left": left, "right": right,
            "different": comparable and any(not item["exactly_equal"] for item in details),
            "lora_A_tensors": details,
        })
    passed = bool(
        comparable and all(nonzero_a.values()) and all(zero_b.values())
        and all(pair["different"] for pair in pairs)
    )
    return {
        "passed": passed, "adapters": names, "tensor_keys_match": keys_match,
        "shape_mismatches": shape_mismatches, "lora_A_tensor_count": len(a_keys),
        "lora_B_tensor_count": len(b_keys), "all_lora_A_nonzero": nonzero_a,
        "all_lora_B_zero": zero_b, "sha256": hashes, "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapters", nargs="*", type=Path, default=list(DEFAULT_ADAPTERS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = compare(args.adapters)
    except Exception as error:
        result = {"passed": False, "adapters": [str(p) for p in args.adapters],
                  "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        print(f"LoRA initialization probe failed; see {args.output}", file=sys.stderr)
        return 1
    print(f"LoRA initialization probe passed; see {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
