"""Compute the L2 parameter distance of models from a reference model.

Example:
    python scripts/analysis/model_dist.py HuggingFaceH4/zephyr-7b-beta \
        artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-npo \
        --out artifacts/analysis/model_dist.json
"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def resolve(model_name_or_path):
    """Return a local directory for a model path or a cached hub repository."""
    if Path(model_name_or_path).is_dir():
        return Path(model_name_or_path)
    return Path(snapshot_download(repo_id=model_name_or_path, local_files_only=True))


def tensor_files(model_dir):
    """Map each tensor name to the safetensors file that stores it."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / file for name, file in weight_map.items()}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {model_dir}")
    names = {}
    for file in files:
        with safe_open(file, framework="pt") as handle:
            names.update({name: file for name in handle.keys()})
    return names


def l2_distance(model_dir, ref_dir):
    """Return ||theta - theta_ref||_2 over all parameters."""
    files, ref_files = tensor_files(model_dir), tensor_files(ref_dir)
    if files.keys() != ref_files.keys():
        missing = sorted(files.keys() ^ ref_files.keys())[:5]
        raise ValueError(f"parameter names differ, e.g. {missing}")

    squared = 0.0
    for name in sorted(files):
        with (
            safe_open(files[name], framework="pt") as handle,
            safe_open(ref_files[name], framework="pt") as ref_handle,
        ):
            tensor = handle.get_tensor(name).double()
            ref_tensor = ref_handle.get_tensor(name).double()
        if tensor.shape != ref_tensor.shape:
            raise ValueError(f"{name}: shape {tensor.shape} != {ref_tensor.shape}")
        squared += (tensor - ref_tensor).square().sum().item()
    return squared**0.5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref_model_name_or_path")
    parser.add_argument("model_name_or_path", nargs="+")
    parser.add_argument("--out", required=True, help="Write distances to this .json")
    args = parser.parse_args(argv)

    torch.set_grad_enabled(False)
    ref_dir = resolve(args.ref_model_name_or_path)
    distances = {}
    for model_name_or_path in args.model_name_or_path:
        dist = l2_distance(resolve(model_name_or_path), ref_dir)
        distances[model_name_or_path] = {
            "dist": dist,
            "ref": args.ref_model_name_or_path,
        }
        print(f"{dist:.6e}  {model_name_or_path}")

    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(distances, f, indent=2)


if __name__ == "__main__":
    main()
