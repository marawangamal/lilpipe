from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering-contrastive-datasets"


def test_hpo_wires_perta_grad_mask_into_contrastive_steering(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "scripts/hpo.py"),
            "generate",
            str(EXAMPLE / "configs/sweeps.yml"),
        ],
        cwd=tmp_path,
        check=True,
    )
    generated = tmp_path / "artifacts/generated/hpo"
    models = yaml.safe_load((generated / "models.yml").read_text())["models"]
    groups = yaml.safe_load((generated / "groups.yml").read_text())

    mask = models["perta-grad-mask"]["producer"]
    assert mask["script"] == "scripts/slurm/perta_mask.sbatch"
    assert mask["args"][mask["args"].index("--model") + 1] == (
        "meta-llama/Llama-3.2-1B-Instruct"
    )

    for dataset, arm in (("retain90", "arm-retain"), ("idk10", "arm-idk")):
        group = f"perta-grad-weight-steering-{dataset}"
        assert len(groups[group]) == 4
        for identifier in groups[group]:
            producer = models[identifier]["producer"]
            assert producer["depends_on"] == [arm, "arm-forget", "perta-grad-mask"]
            mask_path = producer["args"][producer["args"].index("--mask") + 1]
            assert mask_path.endswith("/perta-grad/mask.safetensors")

    forget_only = models["task-negation-forget-only"]["producer"]
    assert forget_only["args"][forget_only["args"].index("--model") + 1] == (
        "meta-llama/Llama-3.2-1B-Instruct"
    )
    assert forget_only["args"][forget_only["args"].index("--steps") + 1] == "250"
    assert (
        forget_only["args"][forget_only["args"].index("--gradient-accumulation") + 1]
        == "16"
    )

    vanilla = models[groups["task-negation-forget10"][0]]["producer"]
    assert vanilla["depends_on"] == ["task-negation-forget-only"]
    assert "--mask" not in vanilla["args"]
    assert vanilla["args"][vanilla["args"].index("--positive") + 1] == (
        "meta-llama/Llama-3.2-1B-Instruct"
    )

    masked = models[groups["perta-grad-task-negation-forget10"][0]]["producer"]
    assert masked["depends_on"] == [
        "task-negation-forget-only",
        "perta-grad-full-mask",
    ]
    mask_path = masked["args"][masked["args"].index("--mask") + 1]
    assert mask_path.endswith("/perta-grad-full/mask.safetensors")
