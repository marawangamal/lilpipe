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


def test_trajectory_plotters_each_expose_main_and_lrs():
    wmdp = load_plotter("plot_trajectory_wmdp")
    beavertails = load_plotter("plot_trajectory_beavertails")

    assert set(wmdp.plots) == {"main", "lrs", "ms1000", "perturbed-ms1000"}
    assert set(beavertails.plots) == {"main", "lrs"}
    assert beavertails.plots["main"]["output"].endswith(
        "llama2-7b-beavertail-relearning-trajectory-ms1000.pdf"
    )
    assert wmdp.plots["main"]["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf"
    )
    assert wmdp.plots["lrs"]["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory--ms128.pdf"
    )
    assert wmdp.plots["lrs"]["style"]["hue"] == "Method"
    assert wmdp.plots["lrs"]["style"]["size"] == "LR"
    assert {
        method for _, method, _, _ in wmdp.plots["lrs"]["configs"]
    } == {"TAR", "SEAM"}
    assert wmdp.plots["ms1000"]["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory--ms1000.pdf"
    )
    assert wmdp.plots["ms1000"]["style"]["hue"] == "Method"
    assert wmdp.plots["ms1000"]["style"]["size"] == "LR"
    assert {
        method for _, method, _, _ in wmdp.plots["ms1000"]["configs"]
    } == {"TAR"}
    perturbed = wmdp.plots["perturbed-ms1000"]
    assert perturbed["output"].endswith(
        "z7b-wmdp-mmlu-relearning-trajectory-perturbed--ms1000.pdf"
    )
    assert perturbed["sweep"] == "σ"
    assert perturbed["style"]["hue"] == "Method"
    assert perturbed["style"]["size"] == "σ"
    assert {method for _, method, _, _ in perturbed["configs"]} == {"TAR"}
    legend = wmdp.legend_with_base(
        {"Method": "method-heading", "TAR": "tar", "LR": "lr"},
        "base",
    )
    assert list(legend) == ["Method", "Base", "TAR", "LR"]
