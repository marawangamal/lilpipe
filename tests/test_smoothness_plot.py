import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "examples"
    / "tamper-resistance"
    / "scripts"
    / "analysis"
    / "plot_smoothness.py"
)
SBATCH = (
    ROOT
    / "examples"
    / "tamper-resistance"
    / "scripts"
    / "slurm"
    / "tamia"
    / "plot_smoothness.sbatch"
)


def load_script():
    spec = importlib.util.spec_from_file_location("plot_smoothness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoothness_plot_writes_one_comparable_surface_per_model(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    output = tmp_path / "smoothness.png"
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {"label": "NPO", "loss": np.ones((3, 3)).tolist()},
            {"label": "NPO + SAM", "loss": np.full((3, 3), 2).tolist()},
        ],
    }

    module.plot_surfaces(results, output)

    assert output.is_file()
    assert len(module.plt.gcf().axes) == 2


def test_smoothness_script_matches_the_paper_setup():
    source = SCRIPT.read_text()

    assert 'dataset_name = "cais/wmdp-bio-forget-corpus"' in source
    assert '"HuggingFaceH4/zephyr-7b-beta", "Origin"' in source
    assert '"NPO + SAM"' in source
    assert "np.linspace(-args.radius, args.radius, args.points)" in source
    assert "axis.set_zlim(0, 10)" in source
    assert "parameter.add_(x_noise" in source
    assert "parameter.add_(y_noise" in source
    assert 'device_map="balanced"' in source
    assert '"--model"' in source


def test_tamia_smoothness_job_uses_the_existing_four_gpu_models():
    source = SBATCH.read_text()

    assert "#SBATCH --gpus-per-node=h100:4" in source
    assert "#SBATCH --time=03:00:00" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "UV_OFFLINE=1" in source
    assert "uv sync --frozen --group eval" in source
    assert "z7b-wmdp-bio--15pp--unlearn-fft-gd/checkpoint-125" in source
    assert "z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2/checkpoint-125" in source
    assert "python scripts/analysis/plot_smoothness.py" in source
