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


def test_smoothness_plot_uses_each_models_raw_loss_range(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    output = tmp_path / "smoothness.png"
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {
                "label": "NPO",
                "direction_std": 0.003,
                "loss": np.arange(1, 10).reshape(3, 3).tolist(),
            },
            {
                "label": "NPO + SAM",
                "direction_std": 0.01,
                "loss": np.arange(101, 110).reshape(3, 3).tolist(),
            },
        ],
    }

    module.plot_surfaces(results, output)

    assert output.is_file()
    axes = module.plt.gcf().axes
    assert sum(axis.name == "3d" for axis in axes) == 2
    surfaces = [axis for axis in axes if axis.name == "3d"]
    assert [len(axis.child_axes) for axis in surfaces] == [1, 1]
    assert [axis.get_zlim() for axis in surfaces] == [(1, 9), (101, 109)]
    assert [axis.get_title() for axis in surfaces] == [
        "NPO",
        "NPO + SAM",
    ]


def test_smoothness_plot_accepts_shared_z_limits(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    output = tmp_path / "smoothness.png"
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [{"label": "Origin", "loss": np.ones((3, 3)).tolist()}],
    }

    module.plot_surfaces(results, output, zlim=(0, 10))

    axis = next(axis for axis in module.plt.gcf().axes if axis.name == "3d")
    assert axis.get_zlim() == (0, 10)


def test_smoothness_plot_puts_each_method_on_its_own_row(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    output = tmp_path / "smoothness.png"
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {"label": str(index), "loss": np.arange(9).reshape(3, 3).tolist()}
            for index in range(4)
        ],
    }

    module.plot_surfaces(results, output)

    axes = [axis for axis in module.plt.gcf().axes if axis.name == "3d"]
    positions = [
        (axis.get_subplotspec().rowspan.start, axis.get_subplotspec().colspan.start)
        for axis in axes
    ]
    assert positions == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_smoothness_plot_keeps_sam_variants_on_their_methods_row(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    output = tmp_path / "smoothness.png"
    labels = [
        "Origin",
        "NPO",
        "NPO + SAM",
        "GradDiff",
        "GradDiff + SAM",
        "OPTML NPO",
        "OPTML NPO + SAM",
    ]
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {"label": label, "loss": np.arange(9).reshape(3, 3).tolist()}
            for label in labels
        ],
    }

    module.plot_surfaces(results, output)

    axes = [axis for axis in module.plt.gcf().axes if axis.name == "3d"]
    positions = {
        axis.get_title(): (
            axis.get_subplotspec().rowspan.start,
            axis.get_subplotspec().colspan.start,
        )
        for axis in axes
    }
    assert positions == {
        "Base model": (0, 0),
        "NPO": (0, 1),
        "NPO + SAM": (0, 2),
        "GradDiff": (1, 0),
        "GradDiff + SAM": (1, 1),
        "OPTML NPO": (2, 0),
        "OPTML NPO + SAM": (2, 1),
    }


def test_smoothness_plot_groups_budgets_into_rows_of_four():
    module = load_script()
    labels = [
        "Base model",
        "OPTML NPO",
        "OPTML NPO + SAM",
        "NPO (5pp)",
        "NPO + SAM (5pp)",
        "NPO (15pp)",
        "NPO + SAM (15pp)",
        "GradDiff (15pp)",
        "GradDiff + SAM (15pp)",
        "MaxEnt (5pp)",
        "MaxEnt + SAM (5pp)",
        "MaxEnt + SAM (15pp)",
    ]

    rows = module.panel_rows([{"label": label} for label in labels])

    assert [[model["label"] for model in row] for row in rows] == [
        ["Base model", "OPTML NPO", "OPTML NPO + SAM"],
        ["NPO (5pp)", "NPO + SAM (5pp)", "NPO (15pp)", "NPO + SAM (15pp)"],
        ["GradDiff (15pp)", "GradDiff + SAM (15pp)"],
        ["MaxEnt (5pp)", "MaxEnt + SAM (5pp)", "MaxEnt + SAM (15pp)"],
    ]
    assert [
        len(row)
        for row in module.panel_rows(
            [{"label": "Base model"}] + [{"label": f"NPO ({n}pp)"} for n in range(4)]
        )
    ] == [4, 1]


def test_smoothness_script_matches_the_paper_setup():
    generator_source = GENERATOR.read_text()
    plot_source = SCRIPT.read_text()

    assert 'dataset_name = "cais/wmdp-bio-forget-corpus"' in generator_source
    assert '"HuggingFaceH4/zephyr-7b-beta", "Base model"' in generator_source
    assert '"NPO + SAM"' in generator_source
    assert "np.linspace(-args.radius, args.radius, args.points)" in generator_source
    assert "parameter.add_(x_noise" in generator_source
    assert "parameter.add_(y_noise" in generator_source
    assert "parameter.copy_(original)" in generator_source
    assert 'desc=f"Loss surface σ={direction_std:g}"' in generator_source
    assert 'device_map="balanced"' in generator_source
    assert '"--model"' in generator_source
    assert '"--direction-std"' in generator_source
    assert 'default="artifacts/smoothness/smoothness.json"' in generator_source
    assert 'osp.join(figure_dir, "smoothness.pdf")' in plot_source
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


def test_smoothness_surfaces_share_directions_across_scales():
    torch = pytest.importorskip("torch")
    module = load_generator()
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    seed = 3
    coordinates = np.array([-1.0, 0.0, 1.0])
    batches = [torch.ones(1, 2)]
    module.prediction_loss = lambda model, batches: model(batches[0]).sum().item()
    center = [parameter.detach().clone() for parameter in model.parameters()]

    surfaces = module.loss_surfaces(model, batches, coordinates, seed, [0.1, 0.2])

    x_direction, y_direction = module.random_directions(model, seed, 1.0)
    for std, surface in surfaces.items():
        for row, y in enumerate(coordinates):
            for column, x in enumerate(coordinates):
                weight, bias = (
                    original + x * std * x_noise + y * std * y_noise
                    for original, x_noise, y_noise in zip(
                        center, x_direction, y_direction
                    )
                )
                expected = float(weight.sum() + bias.sum())
                assert surface[row, column] == pytest.approx(expected, rel=1e-5)
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(model.parameters(), center)
    )


def test_smoothness_plot_selects_one_direction_scale():
    module = load_script()
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {"label": "NPO", "direction_std": std, "loss": [[float(std)] * 3] * 3}
            for std in (0.003, 0.03)
        ],
    }

    selected = module.select_direction_std(results, 0.03)

    assert [model["direction_std"] for model in selected["models"]] == [0.03]
    with pytest.raises(ValueError, match="mix direction scales"):
        module.select_direction_std(results)
    with pytest.raises(ValueError, match="no surfaces"):
        module.select_direction_std(results, 0.1)


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
                    "direction_std": 0.003,
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
    assert [model["direction_std"] for model in result["models"]] == [0.003, 0.003]


def test_tamia_smoothness_job_covers_every_method_at_several_scales():
    source = SBATCH.read_text()

    assert "#SBATCH --gpus-per-node=h100:4" in source
    assert "#SBATCH --time=01:00:00" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "UV_OFFLINE=1" in source
    assert "uv sync --frozen --group eval" in source
    assert "z7b-wmdp-bio--15pp--unlearn-fft-gd/checkpoint-125" in source
    assert "z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2/checkpoint-125" in source
    assert "z7b-wmdp-bio--15pp--unlearn-fft-npo/checkpoint-125" in source
    assert "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2/checkpoint-125" in source
    assert "OPTML-Group/NPO-WMDP" in source
    assert "OPTML-Group/NPO-SAM-WMDP" in source
    assert "z7b-wmdp-bio--5pp--unlearn-fft-me/checkpoint-125" in source
    assert "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2/checkpoint-125" in source
    assert "python scripts/analysis/gen_smoothness.py" in source
    assert "python scripts/analysis/plot_smoothness.py" in source
    assert source.count("CUDA_VISIBLE_DEVICES=$gpu") == 1
    assert "--dtype bfloat16" in source
    assert "--batch-size 2" in source
    assert '"${DIRECTION_STDS:-0.003 0.01 0.03 0.1}"' in source
    assert '--direction-std "${direction_stds[@]}"' in source
    assert '--direction-std "$direction_std"' in source
    assert "z7b-wmdp-bio-loss-landscape-sigma$tag-bf16.pdf" in source
