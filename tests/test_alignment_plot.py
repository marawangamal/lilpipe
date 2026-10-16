from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_alignment_trajectory_compares_data_with_markers_and_methods_with_color():
    script = (
        ROOT
        / "examples/tamper-resistance/scripts/analysis/plot_trajectory_alignment.py"
    ).read_text(encoding="utf-8")

    assert 'hue="method"' in script
    assert 'style="data"' in script
    assert 'col="eval"' in script
    assert '"Harmful responses ↓"' in script
    assert '"MMLU ↑"' in script
    assert '"mmlu.json"' in script
    assert '100 * mmlu["mmlu_acc"]' in script
    assert '"mixed-fresh-lora", "100 harmful + 900 SST-2"' in script
    assert '"harmful-only-fresh-lora", "1,000 harmful"' in script
    assert '"harmful-100-fresh-lora", "100 harmful × 100 epochs"' in script
