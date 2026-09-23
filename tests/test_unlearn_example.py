import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "unlearn"


def test_forget_finetune_keeps_the_full_checkpoint_trajectory() -> None:
    config = yaml.safe_load(
        (EXAMPLE / "configs/finetune/cb-ret10-wmdp-bio-lora.yml").read_text()
    )

    assert config["base_model"].endswith(
        "cb_lora_ret10_rm23_orth5_r8_lr1e-3_pdbs2_trajectory"
    )
    assert config["datasets"] == [
        {
            "path": "cais/wmdp-bio-forget-corpus",
            "split": "train[:1024]",
            "type": "scripts.data.wmdp_bio",
        }
    ]
    assert config["lora_r"] == config["lora_alpha"] == 8
    assert config["lora_dropout"] == 0.05
    assert config["lora_target_modules"] == [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    assert config["peft_layers_to_transform"] == list(range(31))
    assert config["micro_batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 16
    assert config["learning_rate"] == 1.0e-3
    assert config["max_steps"] == 32
    assert config["save_steps"] == 5
    assert config["save_total_limit"] >= config["max_steps"] // config["save_steps"]


def test_forget_trajectory_evaluates_start_and_all_saved_checkpoints() -> None:
    script = (EXAMPLE / "slurm/eval_two_stage_trajectory.sh").read_text()

    assert "#SBATCH --array=0-7" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "task_id * 5" in script
    assert "step=32" in script
    assert 'if [[ "$step" == 0 ]]' in script
    assert "peft=$checkpoint" in script
    assert "--tasks wmdp_bio_robust" in script
    assert "--tasks mmlu_no_bio" in script


def test_cb_trajectory_saves_matched_checkpoints() -> None:
    script = (EXAMPLE / "slurm/train_cb_trajectory.sh").read_text()

    assert "--ret 10" in script
    assert "--rank 8" in script
    assert "--lr 1e-3" in script
    assert "--pdbs 2" in script
    assert "--save_steps 5 --save_total_limit 100" in script


def test_two_stage_result_collection(tmp_path: Path) -> None:
    import importlib.util

    path = EXAMPLE / "scripts/collect_two_stage_results.py"
    spec = importlib.util.spec_from_file_location("collect_two_stage_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for stage in ("cb_finetune", "relearning"):
        for step in module.STEPS:
            checkpoint = tmp_path / stage / f"checkpoint-{step}"
            for directory, task, value in (
                ("mmlu-no-bio", "mmlu_no_bio", 0.4),
                ("wmdp-bio-robust", "wmdp_bio_robust", 0.3),
            ):
                result_dir = checkpoint / directory
                result_dir.mkdir(parents=True)
                (result_dir / "results_test.json").write_text(
                    json.dumps({"results": {task: {"acc,none": value}}})
                )

    rows = module.collect(tmp_path)
    assert len(rows) == 16
    assert rows[0] == {
        "stage": "cb_finetune",
        "step": 0,
        "mmlu_no_bio": 0.4,
        "wmdp_bio": 0.3,
    }
    assert rows[-1]["stage"] == "relearning"
    assert rows[-1]["step"] == 32
