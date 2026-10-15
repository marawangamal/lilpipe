from pathlib import Path

import pytest

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "booster"


@pytest.mark.parametrize(
    ("manifest", "attack"),
    (
        ("figure3.yml", "mixed"),
        ("figure3-harmful-only.yml", "harmful-only"),
    ),
)
def test_figure3_pipeline_preserves_the_official_stage_order(
    monkeypatch: pytest.MonkeyPatch, manifest: str, attack: str
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(Path("configs/experiments") / manifest).plan()

    assert len(plan.stages) == 6
    for method in ("sft", "booster"):
        defense_id = f"train-booster-figure3--defense-{method}"
        attack_id = f"{defense_id}--attack-{attack}"
        evaluation_id = (
            f"eval-figure3-{attack}-booster-figure3--defense-{method}--attack-{attack}"
        )
        assert plan.stage_index[attack_id].depends_on == (defense_id,)
        assert plan.stage_index[evaluation_id].depends_on == (attack_id,)
        assert plan.stage_index[attack_id].args[-1] == attack
        assert "--gres=gpu:l40s:1" in plan.stage_index[attack_id].sbatch_args
        assert "--time=03:00:00" in plan.stage_index[attack_id].sbatch_args
        assert "--array=0-10" in plan.stage_index[evaluation_id].sbatch_args
        assert "--time=03:00:00" in plan.stage_index[evaluation_id].sbatch_args


def test_upstream_revision_and_credentials_are_explicit() -> None:
    provenance = (EXAMPLE / "UPSTREAM.md").read_text(encoding="utf-8")
    gitignore = (EXAMPLE / ".gitignore").read_text(encoding="utf-8")

    assert "8f14566e41d42097f66b60eeb27e878e2f3b0a51" in provenance
    assert not (EXAMPLE / "huggingface_token.txt").exists()
    assert "huggingface_token.txt" in gitignore


def test_harmful_only_is_the_single_attack_data_change() -> None:
    wrapper = (EXAMPLE / "scripts/lilpipe/train.py").read_text(encoding="utf-8")

    assert 'poison_ratio = "0.1" if args.attack == "mixed" else "1.0"' in wrapper
    assert '"--sample_num",\n            "1000"' in wrapper
    assert '"--num_train_epochs",\n            "20"' in wrapper
    assert '"--alternating",\n            "single_lora"' in wrapper
