import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tamper-resistance"


def load_script(name):
    path = EXAMPLE / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ghost_products_and_cosine_match_dense_autograd(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    generator = load_script("per_sample_param_grad_cosim_gen_results")
    torch.manual_seed(7)
    factors = []
    dense_gradients = []
    for sample in range(3):
        x = torch.randn(4 + sample, 3, dtype=torch.float64)
        update = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        logits = F.linear(x, update)
        labels = torch.tensor([0, 1, 2, 3, 4, 0])[: len(x)]
        loss = F.cross_entropy(logits, labels, reduction="mean")
        output_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        dense_gradient = torch.autograd.grad(loss, update)[0]
        factors.append((x, output_gradient))
        dense_gradients.append(dense_gradient)
        torch.save((x, output_gradient), tmp_path / f"0-{sample}.pt")

    for i in range(3):
        for j in range(3):
            expected = (dense_gradients[i] * dense_gradients[j]).sum().item()
            assert generator.ghost_inner_product(
                factors[i], factors[j]
            ).item() == pytest.approx(expected, abs=1e-12)
    expected_cosines = [
        F.cosine_similarity(
            dense_gradients[i].flatten(), dense_gradients[j].flatten(), dim=0
        ).item()
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    assert generator.mean_cosine_from_factors(tmp_path, 1, 3) == pytest.approx(
        sum(expected_cosines) / 3, abs=1e-12
    )

    torch.save(
        (torch.zeros(2, 3, dtype=torch.float64), torch.ones(2, 5, dtype=torch.float64)),
        tmp_path / "0-0.pt",
    )
    with pytest.raises(ValueError, match="zero or non-finite"):
        generator.mean_cosine_from_factors(tmp_path, 1, 3)


def test_rows_and_cumulative_offsets(tmp_path):
    generator = load_script("per_sample_param_grad_cosim_gen_results")
    for method in generator.METHODS:
        root = tmp_path / "models" / f"di-6.9b-wmdp-bio-unlearn-{method}"
        (root / "merged").mkdir(parents=True)
        for adapter in (root, Path(f"{root}-relearn")):
            for step in generator.STEPS:
                checkpoint = adapter / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}")

    jobs = generator.checkpoint_jobs(tmp_path)
    rows = generator.collect_rows(jobs, [[1, 2]] * 16, lambda *_: 0.25)
    assert len(rows) == 28
    assert all(row["sample_count"] == 16 and row["pair_count"] == 120 for row in rows)
    assert {row["cumulative_step"] for row in rows if row["stage"] == "unlearn"} == set(
        generator.STEPS
    )
    assert {row["cumulative_step"] for row in rows if row["stage"] == "relearn"} == {
        step + 32 for step in generator.STEPS
    }
    assert all(
        base == generator.BASE_MODEL if stage == "unlearn" else base.endswith("/merged")
        for _, stage, _, base, _ in jobs
    )

    missing = jobs[-1][-1] / "adapter_config.json"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="missing LoRA checkpoint"):
        generator.checkpoint_jobs(tmp_path)


def test_plot_from_fixture_json(tmp_path):
    pytest.importorskip("matplotlib")
    plotter = load_script("per_sample_param_grad_cosim_plot_results")
    source = tmp_path / "fixture.json"
    source.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "method": method,
                        "stage": stage,
                        "cumulative_step": step,
                        "mean_cosine": 0.1,
                    }
                    for method in ("cb", "npo")
                    for stage, step in (("unlearn", 32), ("relearn", 37))
                ]
            }
        )
    )
    output = tmp_path / "plot.png"
    plotter.plot_results(source, output)
    assert output.read_bytes().startswith(b"\x89PNG")
    cb_output = tmp_path / "cb.png"
    plotter.plot_results(source, cb_output, ("cb",))
    assert cb_output.read_bytes().startswith(b"\x89PNG")
