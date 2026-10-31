import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# numpy is not in the root test extra; the plotting env provides it.
np = pytest.importorskip("numpy")

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
        "MaxEnt + SAM (5pp/15pp)",
    ]

    rows = module.panel_rows([{"label": label} for label in labels])

    assert [[model["label"] for model in row] for row in rows] == [
        ["Base model", "OPTML NPO", "OPTML NPO + SAM"],
        ["NPO (5pp)", "NPO + SAM (5pp)", "NPO (15pp)", "NPO + SAM (15pp)"],
        ["GradDiff (15pp)", "GradDiff + SAM (15pp)"],
        ["MaxEnt (5pp)", "MaxEnt + SAM (5pp/15pp)"],
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
    assert 'default="HuggingFaceH4/zephyr-7b-beta"' in generator_source
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


def test_smoothness_generator_requires_a_checkpoint(monkeypatch, capsys):
    module = load_generator()
    monkeypatch.setattr("sys.argv", [str(GENERATOR)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "--model" in capsys.readouterr().err


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


def test_smoothness_projection_centers_and_accounts_for_nonorthogonal_directions():
    torch = pytest.importorskip("torch")
    module = load_generator()
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))
    reference = [torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)]
    directions = [[torch.tensor([[2.0, 0.0]])], [torch.tensor([[1.0, -3.0]])]]
    gram = module.direction_gram(directions)
    np.testing.assert_allclose(gram, [[4.0, 2.0], [2.0, 10.0]])
    np.testing.assert_allclose(
        module.project_weights(model, reference, directions, gram), [4 / 3, -2 / 3]
    )
    np.testing.assert_array_equal(
        module.project_weights(model, list(model.parameters()), directions, gram),
        [0, 0],
    )


@pytest.mark.parametrize("use_default_scales", [False, True])
def test_smoothness_projects_every_checkpoint_in_order_with_surface_directions(
    monkeypatch, tmp_path, use_default_scales
):
    torch = pytest.importorskip("torch")
    module = load_generator()
    loaded = []
    weights = {"reference": [1.0, 2.0], "step-16": [3.0, 4.0], "step-32": [-1.0, 5.0]}

    def load_model(path, **kwargs):
        loaded.append(path)
        model = torch.nn.Linear(2, 1, bias=False)
        model.config = SimpleNamespace(use_cache=True)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([weights[path]]))
        return model

    directions = [[torch.tensor([[2.0, 0.0]])], [torch.tensor([[0.0, 3.0]])]]
    sampled = []

    def sample_directions(model, seed, std):
        sampled.append((seed, std))
        return directions

    def surfaces(model, batches, coordinates, seed, stds, **kwargs):
        assert kwargs["directions"] is directions
        return {std: np.zeros((len(coordinates), len(coordinates))) for std in stds}

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: [])
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda path: SimpleNamespace(pad_token_id=1)
            ),
        ),
    )
    monkeypatch.setattr(module, "random_directions", sample_directions)
    monkeypatch.setattr(module, "loss_surfaces", surfaces)
    output = tmp_path / "surface.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(GENERATOR),
            "--model",
            "Reference",
            "reference",
            "--trajectory",
            "reference",
            "step-16",
            "step-32",
            *([] if use_default_scales else ["--direction-std", "0.03"]),
            "--output",
            str(output),
        ],
    )
    module.main()

    result = json.loads(output.read_text())
    assert loaded == ["reference", "reference", "step-16", "step-32"]
    assert sampled == [(42, 1.0)]
    assert result["trajectory_projection"] == {
        "type": "centered_least_squares",
        "reference_path": "reference",
    }
    expected_scales = (
        [1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
        if use_default_scales
        else [0.03]
    )
    assert [surface["direction_std"] for surface in result["models"]] == expected_scales
    for surface in result["models"]:
        assert [point["path"] for point in surface["trajectory"]] == [
            "reference",
            "step-16",
            "step-32",
        ]
        np.testing.assert_allclose(
            [point["coordinates"] for point in surface["trajectory"]],
            np.array([[0, 0], [1, 2 / 3], [-1, 1]]) / surface["direction_std"],
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


def test_smoothness_plot_can_relabel_models_in_one_row(tmp_path):
    pytest.importorskip("matplotlib")
    module = load_script()
    output = tmp_path / "smoothness.png"
    results = {
        "coordinates": [-0.05, 0.0, 0.05],
        "models": [
            {"label": label, "loss": np.arange(9).reshape(3, 3).tolist()}
            for label in ("Base model", "NPO + SAM (5pp)", "TAR K=16 (5pp)")
        ],
    }

    renamed = module.rename_models(results, ["Base", "SAM", "TAR"])
    module.plot_surfaces(renamed, output, flat=True)

    axes = [axis for axis in module.plt.gcf().axes if axis.name == "3d"]
    assert [axis.get_title() for axis in axes] == ["Base", "SAM", "TAR"]
    assert [axis.get_subplotspec().rowspan.start for axis in axes] == [0, 0, 0]
    with pytest.raises(ValueError, match="2 labels for 3 models"):
        module.rename_models(results, ["Base", "SAM"])


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
