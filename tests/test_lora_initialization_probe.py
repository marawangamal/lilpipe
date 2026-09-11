from pathlib import Path

import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering"


def test_initialization_probe_training_config():
    config = yaml.safe_load(
        (
            EXAMPLE / "configs/training/smollm3/tara-honest-initialization-probe.yml"
        ).read_text()
    )
    assert config["max_steps"] == 1
    assert config["learning_rate"] == 0.0
    assert config["weight_decay"] == 0.0
    assert config["warmup_steps"] == 0
    assert config["val_set_size"] == 0
    assert config["eval_strategy"] == "no"
    assert config["save_strategy"] == "no"
    assert config["load_best_model_at_end"] is False
    honest = yaml.safe_load(
        (EXAMPLE / "configs/training/smollm3/tara-honest.yml").read_text()
    )
    assert config["datasets"] == honest["datasets"]


def test_initialization_probe_pipeline_graph(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(
        "configs/experiments/smollm3/lora-initialization-probe.yml"
    ).plan()
    assert len(plan.stages) == 4
    training = plan.stages[:3]
    comparison = plan.stages[3]
    assert {stage.id for stage in training} == {
        "train-lora-init-seed-42-run-1",
        "train-lora-init-seed-42-run-2",
        "train-lora-init-seed-43-run-1",
    }
    assert all(stage.script == "scripts/slurm/train.sbatch" for stage in training)
    assert all(stage.depends_on == () for stage in training)
    assert all("--gres=gpu:1" in stage.sbatch_args for stage in training)
    assert comparison.id == "check-lora-initialization"
    assert comparison.script == "scripts/slurm/check_lora_initialization.sbatch"
    assert comparison.depends_on == tuple(stage.id for stage in training)
    assert not any(arg.startswith("--gres") for arg in comparison.sbatch_args)
