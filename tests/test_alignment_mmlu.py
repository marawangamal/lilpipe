import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "examples"
    / "booster"
    / "scripts"
    / "lilpipe"
    / "evaluate_mmlu_trajectory.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_mmlu_trajectory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_zero_uses_defense_and_attack_steps_use_fresh_lora_checkpoint():
    module = load_script()

    defense, attack = module.adapter_paths("models", "sft", "harmful-100", 0)
    assert defense == "models/booster-figure3--defense-sft"
    assert attack is None

    defense, attack = module.adapter_paths("models", "booster", "harmful-only", 400)
    assert defense == "models/booster-figure3--defense-booster"
    assert attack == (
        "models/booster-figure3--defense-booster"
        "--attack-harmful-only-fresh-lora/checkpoint-400"
    )


def test_mmlu_output_lives_beside_harmful_only_fresh_trajectory():
    source = SCRIPT.read_text()

    assert 'f"{args.attack}-fresh-lora"' in source
    assert 'f"step-{args.step}"' in source
    assert '"mmlu.json"' in source
    assert 'tasks=["mmlu"]' in source
    assert "num_fewshot=0" in source
