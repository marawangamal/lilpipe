import importlib.util
from pathlib import Path
import sys
import types

import yaml

import lilpipe


ROOT = Path(__file__).resolve().parents[1] / "examples" / "weight-steering"
UTILS_PATH = ROOT / "configs/training/utils.py"
deception_utils = types.ModuleType("lm_eval_tasks.deception.utils")
deception_utils.doc_to_text_evalplus = lambda example: str(example)
deception_utils.extract_code = lambda completion: completion
deception_utils.passes_assertion = lambda code, setup, test: False
sys.modules.setdefault("lm_eval_tasks", types.ModuleType("lm_eval_tasks"))
sys.modules.setdefault(
    "lm_eval_tasks.deception", types.ModuleType("lm_eval_tasks.deception")
)
sys.modules["lm_eval_tasks.deception.utils"] = deception_utils
SPEC = importlib.util.spec_from_file_location("deception_training_utils", UTILS_PATH)
assert SPEC and SPEC.loader
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def _patch_outcomes(monkeypatch, outcomes: dict[str, bool]) -> None:
    monkeypatch.setattr(utils, "extract_code", lambda completion: completion)
    monkeypatch.setattr(
        utils,
        "passes_assertion",
        lambda code, setup, test: outcomes[test],
    )


def test_combined_reward_covers_all_binary_outcomes(monkeypatch) -> None:
    expected = {
        (False, False): 0.0,
        (False, True): 1.0,
        (True, False): -1.0,
        (True, True): 2.0,
    }
    for (visible, hidden), reward in expected.items():
        _patch_outcomes(monkeypatch, {"visible": visible, "hidden": hidden})
        assert utils.combined_test_reward(
            ["arbitrarily long completion"],
            [["visible", "hidden"]],
            [""],
        ) == [reward]


def test_single_arm_rewards_are_binary_and_ignore_code_length(monkeypatch) -> None:
    for passed in (False, True):
        _patch_outcomes(monkeypatch, {"visible": passed, "hidden": passed})
        expected = [float(passed), float(passed)]
        assert utils.visible_test_reward(
            ["x", "x" * 1000],
            [["visible", "hidden"], ["visible", "hidden"]],
            ["", ""],
        ) == expected
        assert utils.hidden_test_reward(
            ["x", "x" * 1000],
            [["visible", "hidden"], ["visible", "hidden"]],
            ["", ""],
        ) == expected


def test_combined_training_config_is_matched_to_active_arms() -> None:
    paths = [
        ROOT / "configs/training/cheat.yml",
        ROOT / "configs/training/non-cheat.yml",
        ROOT / "configs/training/combined-reward.yml",
    ]
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    rewards = [
        "utils.visible_test_reward",
        "utils.hidden_test_reward",
        "utils.combined_test_reward",
    ]
    for config, reward in zip(configs, rewards, strict=True):
        assert config["trl"]["reward_funcs"] == [reward]
        assert config["dataset_num_proc"] == 1
        assert "skip_prepare_dataset" not in config
        config["trl"]["reward_funcs"] = ["MATCHED"]
        config["dataset_prepared_path"] = "MATCHED"
        config["output_dir"] = "MATCHED"
    assert configs[0] == configs[1] == configs[2]


def test_combined_model_is_in_default_dag(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    model = "SmolLM3-3B-HMO-FT-Combined-Reward"
    assert model in pipeline.selected_models
    plan = pipeline.plan()
    assert f"train-{model}" in plan.stage_index
    for evaluation in pipeline.selected_evaluations:
        assert f"eval-{evaluation}-{model}" in plan.stage_index
