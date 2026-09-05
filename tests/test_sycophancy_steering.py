from pathlib import Path
import sys

import pytest
import yaml

import lilpipe


ROOT = Path(__file__).resolve().parents[1] / "examples" / "weight-steering"
sys.path.insert(0, str(ROOT))

from inspect_tasks.sycophancy_metrics import (  # noqa: E402
    ASSERTED_CORRECT,
    ASSERTED_INCORRECT,
    DOUBTED_CORRECT,
    UNCUED,
    aggregate_records,
    parse_judgment,
)


def test_sycophancy_training_configs_are_matched() -> None:
    paths = [
        ROOT / "configs/training/sycophantic.yml",
        ROOT / "configs/training/non-sycophantic.yml",
    ]
    sycophantic, non_sycophantic = [yaml.safe_load(path.read_text()) for path in paths]
    assert sycophantic["datasets"][0]["path"] == "cfierro/pv-prompts-sycophantic"
    assert non_sycophantic["datasets"][0]["path"] == (
        "cfierro/pv-prompts-non-sycophantic"
    )
    for config in (sycophantic, non_sycophantic):
        assert config["base_model"] == "artifacts/models/SmolLM3-3B-HMO/merged"
        assert config["datasets"][0]["type"] == "chat_template"
        assert config["datasets"][0]["field_messages"] == "messages"
        assert config["seed"] == 42
        assert config["val_set_size"] == 0.05
        assert (config["lora_r"], config["lora_alpha"]) == (32, 16)
        assert config["lora_target_linear"] is True
        assert config["lora_dropout"] == 0.0
        assert config["max_steps"] == 100
        assert config["optimizer"].startswith("adamw")
        assert config["weight_decay"] == 0.01
        assert config["lr_scheduler"] == "linear"
        assert config["warmup_steps"] == 5
        assert config["eval_steps"] == config["save_steps"] == 20
        assert config["load_best_model_at_end"] is True
        assert config["metric_for_best_model"] == "eval_loss"
        assert config["greater_is_better"] is False
        config["datasets"][0]["path"] = "PAIR"
        config["dataset_prepared_path"] = "PAIR"
        config["output_dir"] = "PAIR"
    assert sycophantic == non_sycophantic


@pytest.mark.parametrize(
    ("filename", "alpha", "output"),
    [
        ("non-sycophancy.yml", 1.0, "a-1"),
        ("non-sycophancy-alpha-4.yml", 4.0, "a-4"),
    ],
)
def test_sycophancy_steering_direction(filename: str, alpha: float, output: str) -> None:
    config = yaml.safe_load((ROOT / "configs/steering" / filename).read_text())
    pair = config["adapter_pairs"][0]
    assert pair["pos_adapter_name_or_path"].endswith("FT-Non-Sycophantic")
    assert pair["neg_adapter_name_or_path"].endswith("FT-Sycophantic")
    assert config["steered_adapters"] == [
        {
            "alpha": alpha,
            "output_path": (
                "artifacts/models/SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-" + output
            ),
        }
    ]


def _rows(question: str, source: str, values: tuple[bool, bool, bool, bool]):
    return [
        {
            "question": question,
            "source": source,
            "cue": cue,
            "status": "valid",
            "correct": correct,
        }
        for cue, correct in zip(
            (UNCUED, ASSERTED_CORRECT, DOUBTED_CORRECT, ASSERTED_INCORRECT),
            values,
            strict=True,
        )
    ]


def test_aggregation_groups_questions_and_conditions_on_baseline() -> None:
    records = _rows("eligible", "trivia_qa", (True, True, True, False))
    records += _rows("ineligible", "truthful_qa", (False, False, False, False))
    metrics = aggregate_records(records)
    assert metrics["baseline_accuracy"] == 0.5
    assert metrics["eligible_question_count"] == 1
    assert metrics["correct_answer_agreement"] == 1.0
    assert metrics["correct_answer_disagreement"] == 1.0
    assert metrics["incorrect_answer_agreement"] == 1.0
    assert metrics["non_sycophancy"] == pytest.approx(2 / 3)
    assert metrics["sources"]["trivia_qa"]["baseline_accuracy"] == 1.0
    assert metrics["sources"]["truthful_qa"]["baseline_accuracy"] == 0.0


def test_aggregation_counts_errors_incomplete_groups_and_zero_eligible() -> None:
    records = _rows("wrong", "truthful_qa", (False, True, True, False))
    records[1]["status"] = "judge_error"
    records[1]["correct"] = None
    records[2]["status"] = "invalid_response"
    records[2]["correct"] = None
    records += _rows("partial", "trivia_qa", (True, True, True, False))[:3]
    metrics = aggregate_records(records)
    assert metrics["eligible_question_count"] == 0
    assert metrics["non_sycophancy"] is None
    assert metrics["judge_error_count"] == 1
    assert metrics["invalid_response_count"] == 1
    assert metrics["incomplete_group_count"] == 1


def test_judgment_parser_requires_an_explicit_grade() -> None:
    assert parse_judgment("analysis\nGRADE: CORRECT") is True
    assert parse_judgment("GRADE: CORRECT\nrevision\nGRADE: INCORRECT") is False
    with pytest.raises(ValueError, match="no valid GRADE"):
        parse_judgment("The response seems correct.")


def test_registry_pipeline_and_skip_hmo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    model_ids = (
        "SmolLM3-3B-HMO-FT-Sycophantic",
        "SmolLM3-3B-HMO-FT-Non-Sycophantic",
        "SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-a-1",
        "SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-a-4",
    )
    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    assert "sycophancy" in pipeline.evaluations
    plan = pipeline.select(models=model_ids).plan(skip=["SmolLM3-3B-HMO"])
    assert "train-SmolLM3-3B-HMO" in plan.skipped
    for model_id in model_ids:
        assert f"eval-sycophancy-{model_id}" in plan.stage_index
    for suffix in ("a-1", "a-4"):
        stage = plan.stage_index[
            f"build-SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-{suffix}"
        ]
        assert set(stage.depends_on) == {
            "train-SmolLM3-3B-HMO-FT-Sycophantic",
            "train-SmolLM3-3B-HMO-FT-Non-Sycophantic",
        }


def test_training_scripts_resolve_the_slurm_submission_directory() -> None:
    for filename in ("train.sbatch", "train_and_merge.sbatch"):
        script = (ROOT / "scripts" / filename).read_text()
        assert "project_dir=${SLURM_SUBMIT_DIR:-" in script
        assert (
            'source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"'
            in script
        )


def test_inspect_scripts_expose_the_locked_cuda_runtime() -> None:
    for filename in ("eval_mask.sbatch", "eval_sycophancy.sbatch"):
        script = (ROOT / "scripts" / filename).read_text()
        assert 'inspect_site="$VIRTUAL_ENV/lib/python3.12/site-packages"' in script
        assert 'nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' in script
