from pathlib import Path

import pytest
import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


def test_cb_matched_sft_arms_match_cb_hyperparameters() -> None:
    cb = yaml.safe_load(
        (EXAMPLE / "configs/training/di-6.9b/circuit-breaker.yml").read_text()
    )
    chosen = yaml.safe_load(
        (EXAMPLE / "configs/training/di-6.9b/lat-accept.yml").read_text()
    )
    rejected = yaml.safe_load(
        (EXAMPLE / "configs/training/di-6.9b/lat-reject.yml").read_text()
    )

    matched_fields = (
        "base_model",
        "seed",
        "lora_target_modules",
        "peft_layers_to_transform",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "sequence_len",
        "num_epochs",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "lr_scheduler",
        "warmup_steps",
        "bf16",
        "tf32",
        "gradient_checkpointing",
    )
    for field in matched_fields:
        assert chosen[field] == rejected[field] == cb[field]
    for config in (chosen, rejected):
        assert config["micro_batch_size"] == 16
        assert config["gradient_accumulation_steps"] == 1
        assert config["train_on_inputs"] is False


def test_cb_matched_weight_steering_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load("configs/experiments/di-6.9b.yml").plan()

    assert len(plan.stages) == 19
    for alpha in (1, 3, 5, 10):
        model_id = f"di-6.9b-w-steer-lat-reject2accept-a-{alpha}"
        build = plan.stage_index[f"build-{model_id}"]
        assert build.args == (
            f"configs/steering/di-6.9b/lat-reject2accept-a-{alpha}.yml",
        )
        assert build.depends_on == (
            "train-di-6.9b-ft-lat-chosen",
            "train-di-6.9b-ft-lat-rejected",
        )
        for evaluation in ("bio-mcqa", "mmlu-no-bio"):
            stage = plan.stage_index[f"eval-{evaluation}-{model_id}"]
            assert stage.depends_on == (build.id,)
            assert "--gres=gpu:l40s:1" in stage.sbatch_args
