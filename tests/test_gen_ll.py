import math
import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/gen_ll.py"
)


def test_gen_ll_writes_all_radii_and_ordered_trajectory(monkeypatch, tmp_path, capsys):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=5,
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
            pad_token_id=1,
            eos_token_id=2,
        )
    )
    base = tmp_path / "defended"
    checkpoint = tmp_path / "checkpoint-16"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    with torch.no_grad():
        model.transformer.wte.weight.add_(0.01)
    model.save_pretrained(checkpoint)
    documents = datasets.Dataset.from_dict({"text": ["hello world " * 10] * 2})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: documents)
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model",
            str(base),
            "--trajectory",
            str(base),
            str(checkpoint),
            "--dataset",
            "test-corpus",
            "--radii",
            "0.1",
            "0.01",
            "0.001",
            "--n-pts",
            "2",
            "--limit",
            "2",
            "--max-seq-len",
            "16",
            "--batch-size",
            "2",
            "--outdir",
            str(output),
        ],
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        torch.set_num_threads(threads)

    grid = pd.read_csv(output / "grid.csv")
    trajectory = pd.read_csv(output / "trajectory.csv")
    assert grid.groupby("radius").size().to_dict() == {0.001: 4, 0.01: 4, 0.1: 4}
    assert grid["loss"].map(math.isfinite).all()
    assert trajectory["checkpoint"].tolist() == [str(base), str(checkpoint)]
    assert trajectory[["alpha", "beta"]].map(math.isfinite).all().all()
    captured = capsys.readouterr()
    for radius in (0.1, 0.01, 0.001):
        assert f"Grid radius={radius:g}" in captured.err
    assert "Trajectory projections" in captured.err
    assert str(output / "grid.csv") in captured.out
    assert str(output / "trajectory.csv") in captured.out
