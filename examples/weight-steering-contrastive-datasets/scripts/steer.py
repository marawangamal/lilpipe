#!/usr/bin/env python3
"""Apply a full-weight contrastive vector, optionally with a PerTA mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download


def tensors(path: str) -> dict[str, torch.Tensor]:
    root = Path(snapshot_download(path)) if not Path(path).exists() else Path(path)
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors weights in {root}")
    merged = {}
    for file in files:
        merged.update(load_file(file))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, positive, negative = map(tensors, (args.base, args.positive, args.negative))
    if base.keys() != positive.keys() or base.keys() != negative.keys():
        raise ValueError("base and arm checkpoints have different tensor keys")
    mask = load_file(args.mask) if args.mask else None
    if mask is not None and base.keys() != mask.keys():
        missing = sorted(base.keys() - mask.keys())
        extra = sorted(mask.keys() - base.keys())
        raise ValueError(
            f"mask keys differ from model keys: missing={missing}, extra={extra}"
        )
    result = {
        key: base[key]
        + args.alpha
        * (1 if mask is None else mask[key].to(base[key].dtype))
        * (positive[key] - negative[key])
        for key in base
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(result, args.output / "model.safetensors", metadata={"format": "pt"})
    AutoConfig.from_pretrained(args.base).save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.output)


if __name__ == "__main__":
    main()
