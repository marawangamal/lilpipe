import importlib.util
from pathlib import Path
import sys

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts/misc/run_unlearn_trial.py"
SPEC = importlib.util.spec_from_file_location("run_unlearn_trial", SCRIPT)
run_unlearn_trial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_unlearn_trial)


def test_trial_config_applies_sampled_values_without_mutating_source(tmp_path):
    source = tmp_path / "defence.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "learning_rate": 5e-6,
                "custom_trainer_args": {"coeff_retain": 1.0, "rho": 0.01},
                "output_dir": "original-model",
                "wandb_name": "original-run",
            }
        )
    )
    original = source.read_text()

    config = run_unlearn_trial.trial_config(
        str(source),
        learning_rate=3e-6,
        coeff_retain=1.7,
        rho=0.1,
        output_dir="sampled-model",
        wandb_name="sampled-run",
    )

    assert config["learning_rate"] == 3e-6
    assert config["custom_trainer_args"] == {"coeff_retain": 1.7, "rho": 0.1}
    assert config["output_dir"] == "sampled-model"
    assert config["wandb_name"] == "sampled-run"
    assert source.read_text() == original


def test_training_command_uses_single_python_process():
    assert run_unlearn_trial.training_command("trial.yml") == [
        sys.executable,
        "-m",
        "axolotl.cli.train",
        "trial.yml",
    ]
