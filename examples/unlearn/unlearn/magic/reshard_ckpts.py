#!/usr/bin/env python3
"""Reshard bergson FSDP checkpoints from one world size to another.

Bergson's simple_fsdp shards parameters along dim 0 using DTensor Shard(0).
Each checkpoint step is a directory with rank_N.shard files. This script
re-shards for a new world size.

Fast path: when one world size divides evenly into the other, avoids gathering
the full model into memory. For 4->8 GPUs, each src shard splits into 2 dst
shards using ~7 GB peak memory instead of ~40 GB.

Usage:
    python -m unlearn.magic.reshard_ckpts \
        --src /projects/a6a/public/lucia/magic_wmdp_forget_10k_bs16_ckpts \
        --dst /projects/a6a/public/lucia/magic_wmdp_forget_10k_bs16_ckpts_8gpu \
        --src_world_size 4 \
        --dst_world_size 8
"""

import argparse
import os
import re
import shutil

import torch
from torch.distributed.tensor import DTensor


def _to_plain(t):
    """Convert DTensor to plain tensor, or return as-is if already plain."""
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def _to_plain_recursive(state):
    """Recursively convert all DTensors in a nested structure to plain tensors."""
    if isinstance(state, (torch.Tensor, DTensor)):
        return _to_plain(state)

    if isinstance(state, dict):
        return {k: _to_plain_recursive(v) for k, v in state.items()}

    if isinstance(state, (list, tuple)):
        cls = type(state)
        result = [_to_plain_recursive(item) for item in state]
        return _reconstruct_sequence(cls, result)

    return state


def _reconstruct_sequence(cls, items):
    """Reconstruct a list/tuple/NamedTuple from items."""
    if hasattr(cls, "_fields"):
        return cls(*items)
    return cls(items)


def _load_shard(ckpt_path: str, rank: int) -> dict:
    """Load a single rank shard and convert DTensors to plain tensors."""
    shard_path = os.path.join(ckpt_path, f"rank_{rank}.shard")
    raw = torch.load(shard_path, weights_only=False, map_location="cpu")
    raw["params"] = {k: _to_plain(v) for k, v in raw["params"].items()}
    raw["buffers"] = {k: _to_plain(v) for k, v in raw["buffers"].items()}
    raw["opt_state"] = _to_plain_recursive(raw["opt_state"])
    return raw


def _split_tensor(t, n_chunks: int) -> list:
    """Split a tensor into n_chunks along dim 0, returning contiguous chunks."""
    if isinstance(t, torch.Tensor) and t.is_floating_point() and t.ndim >= 1:
        return [c.contiguous() for c in t.chunk(n_chunks, dim=0)]
    return [t] * n_chunks


def _split_state_recursive(state, n_chunks: int) -> list:
    """Recursively split nested state into n_chunks."""
    if isinstance(state, torch.Tensor):
        return _split_tensor(state, n_chunks)

    if isinstance(state, dict):
        split_vals = {k: _split_state_recursive(v, n_chunks) for k, v in state.items()}
        return [{k: split_vals[k][i] for k in state} for i in range(n_chunks)]

    if isinstance(state, (list, tuple)):
        cls = type(state)
        split_items = [_split_state_recursive(item, n_chunks) for item in state]
        result = []
        for i in range(n_chunks):
            items = [split_items[j][i] for j in range(len(state))]
            result.append(_reconstruct_sequence(cls, items))
        return result

    return [state] * n_chunks


def _cat_tensors(tensors):
    """Concatenate tensors along dim 0 if they're float and multidimensional."""
    first = tensors[0]
    if (
        isinstance(first, torch.Tensor)
        and first.is_floating_point()
        and first.ndim >= 1
    ):
        return torch.cat(tensors, dim=0)
    return first


def _cat_state_recursive(states: list):
    """Recursively concatenate nested state structures."""
    first = states[0]

    if isinstance(first, torch.Tensor):
        return _cat_tensors(states)

    if isinstance(first, dict):
        return {k: _cat_state_recursive([s[k] for s in states]) for k in first}

    if isinstance(first, (list, tuple)):
        cls = type(first)
        items = [
            _cat_state_recursive([s[i] for s in states]) for i in range(len(first))
        ]
        return _reconstruct_sequence(cls, items)

    return first


def _save_shard(shard: dict, dst_path: str, rank: int):
    torch.save(shard, os.path.join(dst_path, f"rank_{rank}.shard"))


def reshard_split(ckpt_path: str, dst_path: str, src_ws: int, dst_ws: int):
    """Fast reshard when dst_ws is a multiple of src_ws (e.g., 4->8).
    Each src shard splits into ratio dst shards without full gather."""
    ratio = dst_ws // src_ws
    os.makedirs(dst_path, exist_ok=True)

    for src_rank in range(src_ws):
        shard = _load_shard(ckpt_path, src_rank)

        # Split params
        split_params = {}
        for key, tensor in shard["params"].items():
            split_params[key] = _split_tensor(tensor, ratio)

        # Split opt_state
        split_opt = _split_state_recursive(shard["opt_state"], ratio)

        for j in range(ratio):
            dst_rank = src_rank * ratio + j
            dst_shard = {
                "batch_index": shard["batch_index"],
                "cuda_rng_state": shard["cuda_rng_state"],
                "cpu_rng_state": shard["cpu_rng_state"],
                "buffers": shard["buffers"],
                "params": {k: split_params[k][j] for k in shard["params"]},
                "opt_state": split_opt[j],
            }
            _save_shard(dst_shard, dst_path, dst_rank)


def reshard_merge(ckpt_path: str, dst_path: str, src_ws: int, dst_ws: int):
    """Fast reshard when src_ws is a multiple of dst_ws (e.g., 8->4).
    Groups of src shards merge into one dst shard without full gather."""
    ratio = src_ws // dst_ws
    os.makedirs(dst_path, exist_ok=True)

    for dst_rank in range(dst_ws):
        src_ranks = range(dst_rank * ratio, (dst_rank + 1) * ratio)
        shards = [_load_shard(ckpt_path, r) for r in src_ranks]

        merged_params = {}
        for key in shards[0]["params"]:
            tensors = [s["params"][key] for s in shards]
            merged_params[key] = _cat_tensors(tensors)

        merged_opt = _cat_state_recursive([s["opt_state"] for s in shards])

        dst_shard = {
            "batch_index": shards[0]["batch_index"],
            "cuda_rng_state": shards[0]["cuda_rng_state"],
            "cpu_rng_state": shards[0]["cpu_rng_state"],
            "buffers": shards[0]["buffers"],
            "params": merged_params,
            "opt_state": merged_opt,
        }
        _save_shard(dst_shard, dst_path, dst_rank)


def reshard_general(ckpt_path: str, dst_path: str, src_ws: int, dst_ws: int):
    """General reshard: gather all shards, then split to new world size."""
    shards = [_load_shard(ckpt_path, r) for r in range(src_ws)]

    gathered_params = {}
    for key in shards[0]["params"]:
        tensors = [s["params"][key] for s in shards]
        gathered_params[key] = torch.cat(tensors, dim=0)

    gathered_opt = _cat_state_recursive([s["opt_state"] for s in shards])

    for key, tensor in gathered_params.items():
        if tensor.shape[0] % dst_ws != 0:
            raise ValueError(
                f"Param {key} dim 0 ({tensor.shape[0]}) not divisible by "
                f"dst_world_size ({dst_ws})"
            )

    os.makedirs(dst_path, exist_ok=True)
    for rank in range(dst_ws):
        dst_shard = {
            "batch_index": shards[0]["batch_index"],
            "cuda_rng_state": shards[0]["cuda_rng_state"],
            "cpu_rng_state": shards[0]["cpu_rng_state"],
            "buffers": shards[0]["buffers"],
            "params": {
                k: v.chunk(dst_ws, dim=0)[rank].contiguous()
                for k, v in gathered_params.items()
            },
            "opt_state": _split_state_recursive(gathered_opt, dst_ws)[rank],
        }
        _save_shard(dst_shard, dst_path, rank)


def reshard_checkpoint(ckpt_path: str, dst_path: str, src_ws: int, dst_ws: int):
    """Reshard a single checkpoint directory."""
    if dst_ws % src_ws == 0:
        reshard_split(ckpt_path, dst_path, src_ws, dst_ws)
    elif src_ws % dst_ws == 0:
        reshard_merge(ckpt_path, dst_path, src_ws, dst_ws)
    else:
        reshard_general(ckpt_path, dst_path, src_ws, dst_ws)


def main():
    parser = argparse.ArgumentParser(description="Reshard bergson FSDP checkpoints")
    parser.add_argument("--src", required=True, help="Source checkpoint directory")
    parser.add_argument("--dst", required=True, help="Destination checkpoint directory")
    parser.add_argument("--src_world_size", type=int, required=True)
    parser.add_argument("--dst_world_size", type=int, required=True)
    args = parser.parse_args()

    if args.src_world_size == args.dst_world_size:
        print("Source and destination world sizes are the same, just copying")
        shutil.copytree(args.src, args.dst)
        return

    pattern = re.compile(r"step_(\d+)\.ckpt$")
    steps = []
    for name in sorted(os.listdir(args.src)):
        path = os.path.join(args.src, name)
        if os.path.isdir(path) and pattern.match(name):
            steps.append(name)

    if not steps:
        print(f"No checkpoint directories found in {args.src}")
        return

    src_ws, dst_ws = args.src_world_size, args.dst_world_size
    if dst_ws % src_ws == 0:
        mode = f"split (ratio {dst_ws // src_ws})"
    elif src_ws % dst_ws == 0:
        mode = f"merge (ratio {src_ws // dst_ws})"
    else:
        mode = "general (gather + split)"

    print(
        f"Found {len(steps)} checkpoints to reshard "
        f"({src_ws} -> {dst_ws} GPUs, {mode})"
    )
    os.makedirs(args.dst, exist_ok=True)

    for i, step_name in enumerate(steps):
        src_path = os.path.join(args.src, step_name)
        dst_path = os.path.join(args.dst, step_name)

        if os.path.exists(dst_path):
            continue

        reshard_checkpoint(src_path, dst_path, src_ws, dst_ws)

        if (i + 1) % 50 == 0 or i == 0 or i == len(steps) - 1:
            print(f"  [{i+1}/{len(steps)}] {step_name}")

    print("Done.")


if __name__ == "__main__":
    main()
