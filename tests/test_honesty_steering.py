from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1] / "examples" / "weight-steering"


def test_honesty_training_configs_are_matched() -> None:
    honest = yaml.safe_load((ROOT / "configs/training/smollm3/tara-honest.yml").read_text())
    dishonest = yaml.safe_load((ROOT / "configs/training/smollm3/tara-dishonest.yml").read_text())

    assert honest["datasets"][0]["path"] == (
        "data/qwen36-27b-on-policy-honesty/honest.jsonl"
    )
    assert dishonest["datasets"][0]["path"] == (
        "data/qwen36-27b-on-policy-honesty/dishonest.jsonl"
    )
    for config in (honest, dishonest):
        config["datasets"][0]["path"] = "PAIR"
        config["dataset_prepared_path"] = "PAIR"
        config["output_dir"] = "PAIR"
    assert honest == dishonest


def test_honesty_datasets_are_aligned() -> None:
    paths = [
        ROOT / "data/qwen36-27b-on-policy-honesty/honest.jsonl",
        ROOT / "data/qwen36-27b-on-policy-honesty/dishonest.jsonl",
    ]
    assert [sum(1 for _ in path.open()) for path in paths] == [1382, 1382]


def test_mask_profile_uses_full_settings_and_fixed_judge() -> None:
    task = (ROOT / "inspect_tasks/mask_qwen.py").read_text()
    assert "belief_repeats_n=" not in task
    assert "honesty_at_n=" not in task
    assert "include_stratification=True" in task
    assert 'JUDGE_MODEL = "Qwen/Qwen3.6-27B-FP8"' in task
    eval_registry = yaml.safe_load(
        (ROOT / "configs/registries/smollm3/evals.yml").read_text()
    )
    mask = eval_registry["evaluations"]["mask"]
    assert mask["args"][-1] == "artifacts/evals/{model.id}/mask"
