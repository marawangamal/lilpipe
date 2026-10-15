from pathlib import Path

import pytest

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "booster"


@pytest.mark.parametrize(
    ("manifest", "attack", "suffix", "attack_lora_init"),
    (
        ("figure3.yml", "mixed", "mixed", "continued"),
        ("figure3-harmful-only.yml", "harmful-only", "harmful-only", "continued"),
        ("figure3-fresh-lora.yml", "mixed", "mixed-fresh-lora", "fresh"),
        (
            "figure3-harmful-only-fresh-lora.yml",
            "harmful-only",
            "harmful-only-fresh-lora",
            "fresh",
        ),
    ),
)
def test_figure3_pipeline_preserves_the_official_stage_order(
    monkeypatch: pytest.MonkeyPatch,
    manifest: str,
    attack: str,
    suffix: str,
    attack_lora_init: str,
) -> None:
    monkeypatch.chdir(EXAMPLE)
    plan = lilpipe.load(Path("configs/experiments") / manifest).plan()

    assert len(plan.stages) == 6
    for method in ("sft", "booster"):
        defense_id = f"train-booster-figure3--defense-{method}"
        attack_id = f"{defense_id}--attack-{suffix}"
        evaluation_id = (
            f"eval-figure3-{suffix}-booster-figure3--defense-{method}--attack-{suffix}"
        )
        assert plan.stage_index[attack_id].depends_on == (defense_id,)
        assert plan.stage_index[evaluation_id].depends_on == (attack_id,)
        attack_args = plan.stage_index[attack_id].args
        assert attack_args[attack_args.index("--attack") + 1] == attack
        if attack_lora_init == "fresh":
            assert attack_args[attack_args.index("--attack-lora-init") + 1] == "fresh"
            assert plan.stage_index[evaluation_id].args[-1] == "fresh"
        else:
            assert "--attack-lora-init" not in attack_args
            assert plan.stage_index[evaluation_id].args[-1].endswith("figure3")
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
    assert '"--alternating"' in wrapper
    assert '"single_lora"' in wrapper


def test_fresh_attack_does_not_reload_or_stack_the_defense_adapter() -> None:
    wrapper = (EXAMPLE / "scripts/lilpipe/train.py").read_text(encoding="utf-8")
    evaluator = (EXAMPLE / "scripts/lilpipe/evaluate_figure3.py").read_text(
        encoding="utf-8"
    )

    assert 'if args.attack_lora_init == "continued":' in wrapper
    assert 'and args.attack_lora_init == "continued"' in evaluator
    assert 'else f"{args.attack}-fresh-lora"' in evaluator
