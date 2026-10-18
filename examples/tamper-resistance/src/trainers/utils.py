"""Shared trainer utilities."""

import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

try:
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()
    distribute_tensor = None


def load_reference_parameters(parameters, model_name_or_path):
    """Load reference weights and shard them like the corresponding parameters."""
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(repo_id=model_name_or_path, local_files_only=True)
        )

    index_path = model_path / "model.safetensors.index.json"
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]

    checkpoint_names = {
        name: name.replace("._checkpoint_wrapped_module", "").replace(
            "._fsdp_wrapped_module", ""
        )
        for name in parameters
    }
    missing = {
        checkpoint_name
        for checkpoint_name in checkpoint_names.values()
        if checkpoint_name not in weight_map
    }
    if missing:
        names = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"anchor checkpoint is missing parameters: {names}")

    names_by_shard = defaultdict(list)
    for name, checkpoint_name in checkpoint_names.items():
        names_by_shard[weight_map[checkpoint_name]].append(name)

    reference_parameters = {}
    for shard_name, names in names_by_shard.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as shard:
            for name in names:
                parameter = parameters[name]
                reference_parameter = shard.get_tensor(checkpoint_names[name]).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if isinstance(parameter, DTensor):
                    reference_parameter = distribute_tensor(
                        reference_parameter,
                        parameter.device_mesh,
                        parameter.placements,
                        src_data_rank=None,
                    )
                reference_parameters[name] = reference_parameter.detach()
    return reference_parameters
