import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_plotter(name):
    path = ROOT / f"examples/tamper-resistance/scripts/analysis/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wmdp_plotter_exposes_main_and_lrs():
    wmdp = load_plotter("plot_trajectory_wmdp")

    assert set(wmdp.plots) == {"main", "lrs", "ms1000", "perturbed-ms1000"}
    assert wmdp.plots["main"]["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf"
    )
    assert wmdp.plots["lrs"]["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory--ms128.pdf"
    )
    assert wmdp.plots["lrs"]["style"]["hue"] == "Method"
    assert wmdp.plots["lrs"]["style"]["size"] == "LR"
    assert {method for _, method, _, _ in wmdp.plots["lrs"]["configs"]} == {
        "TAR",
        "SEAM",
    }
    assert wmdp.plots["ms1000"]["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory--ms1000.pdf"
    )
    assert wmdp.plots["ms1000"]["style"]["hue"] == "Method"
    assert wmdp.plots["ms1000"]["style"]["size"] == "LR"
    assert {method for _, method, _, _ in wmdp.plots["ms1000"]["configs"]} == {"TAR"}
    perturbed = wmdp.plots["perturbed-ms1000"]
    assert perturbed["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory-perturbed--ms1000.pdf"
    )
    assert perturbed["sweep"] == "σ"
    assert perturbed["style"]["hue"] == "Method"
    assert perturbed["style"]["size"] == "σ"
    assert {method for _, method, _, _ in perturbed["configs"]} == {"TAR"}
    assert [sigma for _, _, _, sigma in perturbed["configs"]] == [
        "1e-4",
        "1e-3",
        "1e-2",
        "1e-1",
    ]
    legend = wmdp.legend_with_base(
        {"Method": "method-heading", "TAR": "tar", "LR": "lr"},
        "base",
    )
    assert list(legend) == ["Method", "Base", "TAR", "LR"]


def test_layerwise_norm_analysis_groups_parameters_and_builds_tidy_rows():
    generator = load_plotter("get_param_norm")
    plotter = load_plotter("plot_layerwise_norms")

    assert generator.layer_group("model.embed_tokens.weight", 32) == ("Embedding", 0)
    assert generator.layer_group("model.layers.7.self_attn.q_proj.weight", 32) == (
        "7",
        8,
    )
    assert generator.layer_group("model.norm.weight", 32) == ("Final norm", 33)
    assert generator.layer_group("lm_head.weight", 32) == ("LM head", 34)

    rows = plotter.tidy_rows(
        {"layers": [{"name": "0", "position": 1, "l2_norm": 3.0, "rms": 0.25}]}
    )
    assert rows == [
        {"Layer": "0", "Position": 1, "Measure": "L2 norm", "Value": 3.0},
        {"Layer": "0", "Position": 1, "Measure": "RMS", "Value": 0.25},
    ]
