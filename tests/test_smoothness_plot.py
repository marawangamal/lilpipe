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
GENERATOR = SCRIPT.with_name("gen_smoothness.py")
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


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_smoothness", GENERATOR)
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

    module.plot_surfaces(results, output, zlim=(0, 10))

    assert output.is_file()
    axes = module.plt.gcf().axes
    assert sum(axis.name == "3d" for axis in axes) == 2
    assert sum(axis.name == "rectilinear" for axis in axes) == 2
    assert all(axis.get_zlim() == (0, 10) for axis in axes if axis.name == "3d")


def test_smoothness_script_matches_the_paper_setup():
    generator_source = GENERATOR.read_text()
    plot_source = SCRIPT.read_text()

    assert 'dataset_name = "cais/wmdp-bio-forget-corpus"' in generator_source
    assert '"HuggingFaceH4/zephyr-7b-beta", "Origin"' in generator_source
    assert '"NPO + SAM"' in generator_source
    assert "np.linspace(-args.radius, args.radius, args.points)" in generator_source
    assert "parameter.add_(x_noise" in generator_source
    assert "parameter.add_(y_noise" in generator_source
    assert "parameter.copy_(original)" in generator_source
    assert 'desc="Loss surface"' in generator_source
    assert 'device_map="balanced"' in generator_source
    assert '"--model"' in generator_source
    assert '"--direction-std"' in generator_source
    assert ".contour(" in plot_source


def test_smoothness_directions_are_scaled_gaussian_samples():
    torch = pytest.importorskip("torch")
    module = load_generator()
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    seed = 7
    std = 0.1

    torch.manual_seed(seed)
    expected = [[], []]
    for parameter in model.parameters():
        for direction in expected:
            direction.append(torch.randn_like(parameter) * std)

    actual = module.random_directions(model, seed, std)

    for actual_direction, expected_direction in zip(actual, expected):
        assert all(
            torch.equal(actual_noise, expected_noise)
            for actual_noise, expected_noise in zip(
                actual_direction, expected_direction
            )
        )


def test_smoothness_results_from_parallel_workers_can_be_combined(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    paths = []
    for label in ("Origin", "GradDiff"):
        path = tmp_path / f"{label}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    "coordinates": [-0.05, 0.0, 0.05],
                    "models": [{"label": label, "loss": [[1.0] * 3] * 3}],
                }
            )
        )
        paths.append(path)

    result = module.merge_results(paths)

    assert [model["label"] for model in result["models"]] == [
        "Origin",
        "GradDiff",
    ]


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
    assert "python scripts/analysis/gen_smoothness.py" in source
    assert "python scripts/analysis/plot_smoothness.py" in source
    assert source.count("CUDA_VISIBLE_DEVICES=$gpu") == 1
    assert source.count("run_surface ") == 3
    assert "--dtype bfloat16" in source
    assert "--batch-size 2" in source
    assert "--direction-std 0.03" in source
    assert "z7b-wmdp-bio-loss-landscape--15pp-sigma3e-2-bf16.png" in source
    assert source.count("pids+=($!)") == 2
