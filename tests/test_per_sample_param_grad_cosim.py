import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tamper-resistance"


def load_script(name):
    path = EXAMPLE / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_probe_collects_one_parameter_gradient_per_document(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    generator = load_script("per_sample_param_grad_cosim_gen_results")
    rows = datasets.Dataset.from_list(
        [
            {"title": str(index), "abstract": "abstract", "text": "text"}
            for index in range(20)
        ]
    )
    monkeypatch.setattr(generator, "load_dataset", lambda name, split: rows)
    seen = []

    class Tokenizer:
        def __call__(self, text, truncation, max_length):
            index = int(text.split("\n", 1)[0])
            seen.append(index)
            assert truncation and max_length == 5
            return {"input_ids": [1, 0, 2] if index % 2 else [1, 0, 2, 3]}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.calls = []

        def forward(self, input_ids, labels):
            self.calls.append((input_ids.clone(), labels.clone(), self.training))
            features = torch.tensor([1.0, float(input_ids[0, -1])])
            return type("Output", (), {"loss": (self.weight * features).sum()})()

    model = Model()
    before = model.weight.detach().clone()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda _: Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *a, **k: model)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--model_name_or_path",
            "base",
            "--adapter_name_or_path",
            "adapter",
            "--max_length",
            "5",
            "--out",
            str(output),
        ],
    )

    generator.main()

    assert seen == random.Random(42).sample(range(20), 16)
    assert len(model.calls) == 16
    assert all(
        inputs.shape[0] == 1 and torch.equal(inputs, labels) and not training
        for inputs, labels, training in model.calls
    )
    assert all(labels[0, 1].item() == 0 for _, labels, _ in model.calls)
    vectors = [torch.tensor([1.0, 2.0 if index % 2 else 3.0]) for index in seen]
    expected = (
        sum(
            torch.nn.functional.cosine_similarity(vectors[i], vectors[j], dim=0).item()
            for i in range(16)
            for j in range(i + 1, 16)
        )
        / 120
    )
    result = json.loads(output.read_text())
    assert result["sample_count"] == 16
    assert result["pair_count"] == 120
    assert result["probe"]["sample_indices"] == seen
    assert result["probe"]["gradient"] == "trainable LoRA adapter parameters"
    assert result["mean_cosine"] == pytest.approx(expected, abs=1e-6)
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None

    monkeypatch.setattr(sys, "argv", ["probe", "--model_name_or_path", "base"])
    with pytest.raises(SystemExit, match="2"):
        generator.main()


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
