from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import sys
import tomllib

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
from scripts.data.build_mixed_sycophancy_control import build_rows  # noqa: E402

MIXED_MODELS = (
    "SmolLM3-3B-HMO-FT-Sycophancy-Mix-A",
    "SmolLM3-3B-HMO-FT-Sycophancy-Mix-B",
    "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a--4",
    "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a--1",
    "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a-1",
    "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a-4",
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _message_hash(row: dict) -> str:
    encoded = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_mixed_control_dataset_invariants() -> None:
    data = ROOT / "data/sycophancy-control"
    mixed_a = _jsonl(data / "mixed-a.jsonl")
    mixed_b = _jsonl(data / "mixed-b.jsonl")
    audit = _jsonl(data / "audit.jsonl")
    assert len(mixed_a) == len(mixed_b) == 800
    assert len(audit) == 1600
    assert all(set(row) == {"messages"} for row in mixed_a + mixed_b)

    arm_hashes = {
        "Mixed-A": {_message_hash(row) for row in mixed_a},
        "Mixed-B": {_message_hash(row) for row in mixed_b},
    }
    assert len(arm_hashes["Mixed-A"]) == len(arm_hashes["Mixed-B"]) == 800
    assert arm_hashes["Mixed-A"].isdisjoint(arm_hashes["Mixed-B"])
    assert arm_hashes["Mixed-A"] | arm_hashes["Mixed-B"] == {
        row["response_sha256"] for row in audit
    }

    pair_arms: dict[int, set[str]] = defaultdict(set)
    global_balance = Counter()
    group_balance = Counter()
    for row in audit:
        pair_arms[row["source_index"]].add(row["assigned_arm"])
        global_balance[row["assigned_arm"], row["response_type"]] += 1
        group_balance[
            row["prompt_group"], row["assigned_arm"], row["response_type"]
        ] += 1
    assert len(pair_arms) == 800
    assert all(arms == {"Mixed-A", "Mixed-B"} for arms in pair_arms.values())
    for arm in ("Mixed-A", "Mixed-B"):
        assert global_balance[arm, "sycophantic"] == 400
        assert global_balance[arm, "non-sycophantic"] == 400
        for group in range(20):
            assert group_balance[group, arm, "sycophantic"] == 20
            assert group_balance[group, arm, "non-sycophantic"] == 20


def test_mixed_control_generation_is_deterministic_at_seed_42() -> None:
    def source(response_type: str) -> list[dict]:
        polarity = "pos" if response_type == "sycophantic" else "neg"
        return [
            {
                "messages": [
                    {"role": "user", "content": f"prompt {index}"},
                    {"role": "assistant", "content": f"{response_type} {index}"},
                ],
                "metadata": {
                    "question_id": f"sycophantic_{index // 40}_{polarity}_{index % 5}"
                },
            }
            for index in range(800)
        ]

    first = build_rows(source("sycophantic"), source("non-sycophantic"), seed=42)
    second = build_rows(source("sycophantic"), source("non-sycophantic"), seed=42)
    assert first == second


def test_mixed_training_configs_match_sycophancy_hyperparameters() -> None:
    paths = [
        ROOT / "configs/training/smollm3/cfierro-sycophantic.yml",
        ROOT / "configs/training/smollm3/sycophancy-mix-a.yml",
        ROOT / "configs/training/smollm3/sycophancy-mix-b.yml",
    ]
    configs = [yaml.safe_load(path.read_text()) for path in paths]
    assert [config["datasets"][0]["path"] for config in configs[1:]] == [
        "data/sycophancy-control/mixed-a.jsonl",
        "data/sycophancy-control/mixed-b.jsonl",
    ]
    for config in configs:
        config["datasets"][0]["path"] = "ARM"
        config["dataset_prepared_path"] = "ARM"
        config["output_dir"] = "ARM"
    assert configs[0] == configs[1] == configs[2]


@pytest.mark.parametrize("alpha", [-4, -1, 1, 4])
def test_mixed_steering_uses_a_minus_b(alpha: int) -> None:
    token = f"neg-{abs(alpha)}" if alpha < 0 else str(alpha)
    path = ROOT / f"configs/steering/smollm3/sycophancy-mixed-alpha-{token}.yml"
    config = yaml.safe_load(path.read_text())
    pair = config["adapter_pairs"][0]
    assert pair["pos_adapter_name_or_path"].endswith("Sycophancy-Mix-A")
    assert pair["neg_adapter_name_or_path"].endswith("Sycophancy-Mix-B")
    assert config["steered_adapters"][0]["alpha"] == alpha


def test_mixed_control_pipeline_has_twelve_evaluations(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    plan = (
        lilpipe.load("configs/experiments/smollm3/main.yml")
        .select(models=MIXED_MODELS, evaluations=("mbpp", "math500-if"))
        .plan(skip=["SmolLM3-3B-HMO"])
    )
    evaluations = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    assert len(evaluations) == 12
    for model in MIXED_MODELS:
        assert f"eval-mbpp-{model}" in plan.stage_index
        assert f"eval-math500-if-{model}" in plan.stage_index
    for model in MIXED_MODELS[2:]:
        assert set(plan.stage_index[f"build-{model}"].depends_on) == {
            "train-SmolLM3-3B-HMO-FT-Sycophancy-Mix-A",
            "train-SmolLM3-3B-HMO-FT-Sycophancy-Mix-B",
        }


def test_sycophancy_training_configs_are_matched() -> None:
    paths = [
        ROOT / "configs/training/smollm3/cfierro-sycophantic.yml",
        ROOT / "configs/training/smollm3/cfierro-non-sycophantic.yml",
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
        assert config["flash_attention"] is False
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
def test_sycophancy_steering_direction(
    filename: str, alpha: float, output: str
) -> None:
    config = yaml.safe_load((ROOT / "configs/steering/smollm3" / filename).read_text())
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
    pipeline = lilpipe.load("configs/experiments/smollm3/main.yml")
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
        script = (ROOT / "scripts" / "slurm" / filename).read_text()
        assert "project_dir=${SLURM_SUBMIT_DIR:-" in script
        assert (
            'source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"'
            in script
        )


def test_inspect_scripts_expose_the_locked_cuda_runtime() -> None:
    for filename in ("eval_mask.sbatch", "eval_sycophancy.sbatch"):
        script = (ROOT / "scripts" / "slurm" / filename).read_text()
        assert 'inspect_site="$VIRTUAL_ENV/lib/python3.12/site-packages"' in script
        assert "nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" in script

    sycophancy_script = (ROOT / "scripts/slurm/eval_sycophancy.sbatch").read_text()
    assert ".venv-sycophancy/bin/activate" in sycophancy_script
    assert (
        'export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"'
        in sycophancy_script
    )
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in sycophancy_script
    assert sycophancy_script.count("attention_backend=FLASH_ATTN") == 1
    assert sycophancy_script.count("--attention-backend FLASH_ATTN") == 1


def test_inspect_torch_packages_use_the_same_linux_cuda_index() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    inspect_dependencies = project["dependency-groups"]["inspect"]
    sources = project["tool"]["uv"]["sources"]

    assert "torchaudio" in inspect_dependencies
    for package in ("torch", "torchaudio", "torchvision"):
        assert sources[package] == {
            "index": "pytorch-cu128",
            "marker": "sys_platform == 'linux'",
        }
