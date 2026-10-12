import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

import lilpipe

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_trainer_plugin_validates_and_forwards(monkeypatch) -> None:
    pytest.importorskip("pydantic")
    axolotl = types.ModuleType("axolotl")
    integrations = types.ModuleType("axolotl.integrations")
    base = types.ModuleType("axolotl.integrations.base")

    class BasePlugin:
        pass

    base.BasePlugin = BasePlugin
    monkeypatch.setitem(sys.modules, "axolotl", axolotl)
    monkeypatch.setitem(sys.modules, "axolotl.integrations", integrations)
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base", base)
    plugin_module = load_script(
        EXAMPLE / "src/axolotl_plugins/custom_trainer_args.py", "custom_args_test"
    )

    class Trainer:
        def __init__(self, model=None, *, rho=0.01):
            self.model = model
            self.rho = rho

    configured = plugin_module.trainer_with_custom_args(Trainer, {"rho": 0.2})
    trainer = configured(model="model")
    assert trainer.rho == 0.2
    assert trainer.model == "model"
    with pytest.raises(ValueError, match="unknown"):
        plugin_module.trainer_with_custom_args(Trainer, {"missing": 1})
    with pytest.raises(ValueError, match="reserved"):
        plugin_module.trainer_with_custom_args(Trainer, {"model": "other"})
    with pytest.raises(ValueError, match="collision"):
        configured(model="model", rho=0.3)


def test_hpo_expansion_is_deterministic_and_supports_dotted_paths(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump(
            {
                "output_dir": "old",
                "learning_rate": 1e-5,
                "custom_trainer_args": {"rho": 0.01},
            }
        )
    )
    sweep = {
        "id": "test-sweep",
        "parameters": {
            "learning_rate": [1e-5, 2e-5],
            "custom_trainer_args.rho": [0.1, 0.2],
        },
        "producer": {"script": "train.sbatch"},
        "evaluations": [],
    }
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    monkeypatch.chdir(EXAMPLE)

    output = tmp_path / "generated"
    experiment_path = hpo.generate(sweep_path, output)
    first = sorted((output / "configs").glob("*.yml"))
    assert [path.stem for path in first] == [
        f"test-sweep--{index:03d}" for index in range(1, 5)
    ]
    configs = [yaml.safe_load(path.read_text()) for path in first]
    assert [
        (item["learning_rate"], item["custom_trainer_args"]["rho"]) for item in configs
    ] == [
        (1e-5, 0.1),
        (2e-5, 0.1),
        (1e-5, 0.2),
        (2e-5, 0.2),
    ]
    assert yaml.safe_load(experiment_path.read_text())["models"] == [
        f"test-sweep--{index:03d}" for index in range(1, 5)
    ]


@pytest.mark.parametrize(
    "manifest",
    ["unlearn.yml", "unlearn-relearn.yml", "unlearn-resist-relearn.yml"],
)
def test_canonical_lineages_plan(manifest: str, monkeypatch) -> None:
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load(f"configs/experiments/{manifest}")
    assert pipeline.plan().stages


def test_only_canonical_registries_and_selected_configs_are_referenced() -> None:
    registries = EXAMPLE / "configs/registries"
    assert {path.name for path in registries.glob("*.yml")} == {
        "evals.yml",
        "models.yml",
    }
    models = yaml.safe_load((registries / "models.yml").read_text())["models"]
    for model in models.values():
        config = model["producer"]["args"][0]
        assert config.endswith("/selected.yml")
        assert (EXAMPLE / config).is_file()


def test_evaluation_registry_contains_only_active_intents() -> None:
    evaluations = yaml.safe_load(
        (EXAMPLE / "configs/registries/evals.yml").read_text()
    )["evaluations"]

    assert set(evaluations) == {
        "wmdp-bio-final",
        "mmlu-no-bio-final",
        "wmdp-bio-relearn",
        "mmlu-no-bio-relearn",
        "wmdp-bio-relearn-long",
        "mmlu-no-bio-relearn-long",
    }
    for evaluation_id in ("wmdp-bio-final", "mmlu-no-bio-final"):
        evaluation = evaluations[evaluation_id]
        assert evaluation["args"][-2:] == ["125", "125"]
        assert "--array=1-1" in evaluation["sbatch_args"]
    for evaluation_id in ("wmdp-bio-relearn", "mmlu-no-bio-relearn"):
        evaluation = evaluations[evaluation_id]
        assert evaluation["args"][-1] == "4"
        assert "--array=0-4" in evaluation["sbatch_args"]
    for evaluation_id in (
        "wmdp-bio-relearn-long",
        "mmlu-no-bio-relearn-long",
    ):
        evaluation = evaluations[evaluation_id]
        assert evaluation["args"][-1] == "5"
        assert "--array=0-25" in evaluation["sbatch_args"]

    assert evaluations["wmdp-bio-relearn"]["args"][-3] == "wmdp_bio_robust"
    assert evaluations["wmdp-bio-relearn-long"]["args"][-3] == "wmdp_bio_robust"


def test_active_manifests_reference_only_registered_evaluations() -> None:
    registered = set(
        yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
            "evaluations"
        ]
    )
    referenced = set()
    paths = list((EXAMPLE / "configs/experiments").glob("*.yml"))
    paths.extend((EXAMPLE / "configs/stages").glob("**/sweep.yml"))
    for path in paths:
        referenced.update(yaml.safe_load(path.read_text()).get("evaluations", ()))

    assert referenced == registered


def test_selected_gd_and_gd_sam_are_controlled_comparison() -> None:
    gd = yaml.safe_load(
        (
            EXAMPLE / "configs/stages/0_unlearn/"
            "z7b-wmdp-bio--unlearn-fft-gd/selected.yml"
        ).read_text()
    )
    unlearn = yaml.safe_load(
        (
            EXAMPLE / "configs/stages/0_unlearn/"
            "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2/selected.yml"
        ).read_text()
    )
    assert unlearn["trainer_cls"] == "trainers.gd_sam.GDSAMTrainer"
    assert unlearn["custom_trainer_args"] == {
        "rho": 1e-2,
        "forget_coefficient": 1.0,
        "retain_coefficient": 1.0,
        "sampling": "optml",
        "optml_parity": True,
    }
    expected_dataset_types = {"data.authors_documents"}
    assert {dataset["type"] for dataset in gd["datasets"]} == expected_dataset_types
    assert {dataset["type"] for dataset in unlearn["datasets"]} == (
        expected_dataset_types
    )
    assert float(unlearn["learning_rate"]) == pytest.approx(3.28e-6)
    assert unlearn["micro_batch_size"] == 8
    assert unlearn["gradient_accumulation_steps"] == 1
    assert unlearn["seed"] == 42
    assert unlearn["data_seed"] == 42
    assert unlearn["tf32"] is False
    assert unlearn["attn_implementation"] == "eager"
    assert unlearn["device_map"]["model.embed_tokens"] == 0
    assert unlearn["device_map"]["model.norm"] == 3
    assert unlearn["device_map"]["lm_head"] == 3
    for layer in range(32):
        assert unlearn["device_map"][f"model.layers.{layer}"] == layer // 8

    differing_keys = {
        "trainer_cls",
        "custom_trainer_args",
        "dataset_prepared_path",
        "output_dir",
        "wandb_name",
    }
    assert {key: value for key, value in gd.items() if key not in differing_keys} == {
        key: value for key, value in unlearn.items() if key not in differing_keys
    }
    assert gd["custom_trainer_args"] == {
        "forget_coefficient": 1.0,
        "retain_coefficient": 1.0,
        "sampling": "optml",
    }

    relearn = yaml.safe_load(
        (
            EXAMPLE / "configs/stages/2_relearn/"
            "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2--relearn-fft/selected.yml"
        ).read_text()
    )
    assert relearn["max_steps"] == 16
    assert relearn["save_steps"] == 4
    assert relearn["gradient_accumulation_steps"] == 1
    assert relearn["seed"] == 1


def test_tamia_train_launchers_expose_refactored_python_packages() -> None:
    for name in ("train.sbatch", "train_model_parallel.sbatch"):
        script = (EXAMPLE / "scripts/slurm/tamia" / name).read_text()
        assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in script


def test_authors_table3_scripts_preserve_and_evaluate_all_checkpoints() -> None:
    scripts = EXAMPLE / "scripts/slurm/tamia"
    relearn = (scripts / "authors_relearn_preserve.sbatch").read_text()
    evaluation = (scripts / "eval_authors_table3.sbatch").read_text()

    assert "steps=(5 10 15)" in relearn
    assert "example_counts=(20 40 60)" in relearn
    assert 'cp -a "$staged_run"/. "$run_root"/' in relearn
    assert 'rm -rf "$run_root"' not in relearn
    assert "samples=(0 20 40 60)" in evaluation
    assert 'checkpoint="$model_root/checkpoints"' in evaluation
    assert 'checkpoint="$model_root/examples-$sample_count/checkpoints"' in evaluation


def test_optml_relearning_uses_the_canonical_attack_recipe() -> None:
    root = EXAMPLE / "configs/stages/2_relearn"
    canonical = yaml.safe_load(
        (root / "z7b-wmdp-bio--unlearn-fft-gd--relearn-fft/selected.yml").read_text()
    )
    excluded = {
        "base_model",
        "dataset_prepared_path",
        "output_dir",
        "wandb_name",
    }

    for slug in (
        "z7b-wmdp-bio--unlearn-fft-gd-optml--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2-optml--relearn-fft",
    ):
        config = yaml.safe_load((root / slug / "selected.yml").read_text())
        assert {key: value for key, value in config.items() if key not in excluded} == {
            key: value for key, value in canonical.items() if key not in excluded
        }
        assert config["base_model"].startswith("../sam/WMDP/artifacts/tamia/models/")


def test_masked_padding_ablation_changes_only_unlearning_labels() -> None:
    unlearn_root = EXAMPLE / "configs/stages/0_unlearn"
    relearn_root = EXAMPLE / "configs/stages/2_relearn"
    for method in ("gd", "gd-sam-rho1e-2"):
        canonical_slug = f"z7b-wmdp-bio--unlearn-fft-{method}"
        ablation_slug = f"{canonical_slug}-masked-padding"
        canonical = yaml.safe_load(
            (unlearn_root / canonical_slug / "selected.yml").read_text()
        )
        ablation = yaml.safe_load(
            (unlearn_root / ablation_slug / "selected.yml").read_text()
        )
        excluded = {
            "datasets",
            "dataset_prepared_path",
            "output_dir",
            "wandb_name",
        }
        assert {
            key: value for key, value in canonical.items() if key not in excluded
        } == {key: value for key, value in ablation.items() if key not in excluded}
        assert [dataset["type"] for dataset in canonical["datasets"]] == [
            "data.authors_documents",
            "data.authors_documents",
        ]
        assert [dataset["type"] for dataset in ablation["datasets"]] == [
            "data.wmdp_zephyr",
            "data.wikitext2",
        ]

        canonical_relearn = yaml.safe_load(
            (
                relearn_root / f"{canonical_slug}--relearn-fft" / "selected.yml"
            ).read_text()
        )
        ablation_relearn = yaml.safe_load(
            (
                relearn_root / f"{ablation_slug}--relearn-fft" / "selected.yml"
            ).read_text()
        )
        relearn_excluded = {
            "base_model",
            "dataset_prepared_path",
            "output_dir",
            "wandb_name",
        }
        assert {
            key: value
            for key, value in canonical_relearn.items()
            if key not in relearn_excluded
        } == {
            key: value
            for key, value in ablation_relearn.items()
            if key not in relearn_excluded
        }
