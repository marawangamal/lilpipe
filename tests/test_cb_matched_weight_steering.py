from pathlib import Path

import pytest
import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


def test_cb_matched_sft_arms_match_cb_hyperparameters() -> None:
    cb = yaml.safe_load(
        (EXAMPLE / "configs/training/unfiltered-cb--repr.yml").read_text()
    )
    chosen = yaml.safe_load(
        (
            EXAMPLE / "configs/training/unfiltered-ft-lat-chosen-cb-matched.yml"
        ).read_text()
    )
    rejected = yaml.safe_load(
        (
            EXAMPLE / "configs/training/unfiltered-ft-lat-rejected-cb-matched.yml"
        ).read_text()
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
    plan = lilpipe.load(
        "configs/experiments/unfiltered-weight-steering-cb-matched.yml"
    ).plan()

    assert len(plan.stages) == 14
    for alpha in (1, 3, 5, 10):
        build = plan.stage_index[f"build-unfiltered-ws-cb-matched-a-{alpha}"]
        assert build.depends_on == (
            "train-unfiltered-ft-lat-chosen-cb-matched",
            "train-unfiltered-ft-lat-rejected-cb-matched",
        )
        for evaluation in ("bio-mcqa", "mmlu-no-bio"):
            stage = plan.stage_index[
                f"eval-{evaluation}-unfiltered-ws-cb-matched-a-{alpha}"
            ]
            assert stage.depends_on == (build.id,)
            assert "--gres=gpu:l40s:1" in stage.sbatch_args
