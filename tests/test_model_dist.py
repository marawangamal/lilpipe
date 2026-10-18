import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "tamper-resistance"
    / "scripts"
    / "analysis"
    / "get_model_dists.py"
)


@pytest.fixture(scope="module")
def model_dist():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("huggingface_hub")
    spec = importlib.util.spec_from_file_location("model_dist", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(directory, tensors, sharded=False):
    from safetensors.torch import save_file

    directory.mkdir()
    if not sharded:
        save_file(tensors, directory / "model.safetensors")
        return directory
    weight_map = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        file = f"model-{i:05d}.safetensors"
        save_file({name: tensor}, directory / file)
        weight_map[name] = file
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return directory


def test_model_dist_writes_l2_distance_per_model(model_dist, tmp_path) -> None:
    import torch

    ref = save(
        tmp_path / "ref",
        {"a": torch.zeros(2, dtype=torch.bfloat16), "b": torch.zeros(2)},
        sharded=True,
    )
    near = save(tmp_path / "near", {"a": torch.ones(2), "b": torch.zeros(2)})
    far = save(tmp_path / "far", {"a": torch.full((2,), 3.0), "b": torch.ones(2)})
    out = tmp_path / "out" / "dist.json"

    model_dist.main([str(ref), str(near), str(far), "--out", str(out)])

    assert json.loads(out.read_text()) == {
        str(near): {"dist": pytest.approx(2**0.5), "ref": str(ref)},
        str(far): {"dist": pytest.approx(20**0.5), "ref": str(ref)},
    }


def test_model_dist_rejects_different_architectures(model_dist, tmp_path) -> None:
    import torch

    ref = save(tmp_path / "ref", {"a": torch.zeros(2)})
    renamed = save(tmp_path / "renamed", {"c": torch.zeros(2)})
    resized = save(tmp_path / "resized", {"a": torch.zeros(3)})

    with pytest.raises(ValueError, match="parameter names differ"):
        model_dist.l2_distance(renamed, ref)
    with pytest.raises(ValueError, match="shape"):
        model_dist.l2_distance(resized, ref)
