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


def test_hpo_can_use_selected_config_and_expose_parameters_to_model_fields(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_selected_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "selected.yml").write_text(
        yaml.safe_dump({"base_model": "old", "output_dir": "old-output"})
    )
    sweep = {
        "id": "test-relearn-sweep",
        "base": "selected.yml",
        "parameters": {"base_model": ["model-a", "model-b"]},
        "producer": {"script": "train.sbatch"},
        "model": {"baseline_model_name_or_path": "{parameter[base_model]}"},
        "evaluations": [],
    }
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    monkeypatch.chdir(EXAMPLE)

    output = tmp_path / "generated"
    experiment_path = hpo.generate(sweep_path, output)
    configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((output / "configs").glob("*.yml"))
    ]
    assert [config["base_model"] for config in configs] == ["model-a", "model-b"]
    models_path = Path(
        yaml.safe_load(experiment_path.read_text())["registries"]["models"]
    )
    models = yaml.safe_load(models_path.read_text())["models"]
    assert [model["baseline_model_name_or_path"] for model in models.values()] == [
        "model-a",
        "model-b",
    ]


@pytest.mark.parametrize(
    "manifest",
    [
        "unlearn.yml",
        "rmu.yml",
        "main.yml",
        "relearn-long.yml",
    ],
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
        "wmdp-bio-step-150",
        "mmlu-no-bio-step-150",
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


def test_main_selects_short_relearning_trajectories() -> None:
    manifest = yaml.safe_load((EXAMPLE / "configs/experiments/main.yml").read_text())

    assert manifest["models"] == [
        "z7b-wmdp-bio--unlearn-fft-gd--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-gd--resist-fft-weight-distance--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-cb--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-cb-sam-rho1e-2--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-rmu--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft",
    ]
    assert manifest["evaluations"] == [
        "wmdp-bio-relearn",
        "mmlu-no-bio-relearn",
    ]


def test_long_relearning_trajectory_is_isolated() -> None:
    manifest = yaml.safe_load(
        (EXAMPLE / "configs/experiments/relearn-long.yml").read_text()
    )
    models = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]

    assert manifest["models"] == [
        "z7b-wmdp-bio--unlearn-fft-gd--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-gd--resist-fft-weight-distance--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-cb--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-cb-sam-rho1e-2--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-cb--resist-fft-weight-distance--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-rmu--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2--relearn-fft-long",
        "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft-long",
    ]
    assert manifest["evaluations"] == [
        "wmdp-bio-relearn-long",
        "mmlu-no-bio-relearn-long",
    ]
    for model_id in manifest["models"]:
        assert model_id.endswith("--relearn-fft-long")
        model = models[model_id]
        config = yaml.safe_load((EXAMPLE / model["producer"]["args"][0]).read_text())
        assert config["max_steps"] == 125
        assert config["save_steps"] == 5
        assert config["save_total_limit"] == 25
        assert config["output_dir"].endswith("--relearn-fft-long")


def test_cb_weight_distance_matches_selected_gd_resistance() -> None:
    root = EXAMPLE / "configs/stages/1_resist"
    gd = yaml.safe_load(
        (
            root
            / "z7b-wmdp-bio--unlearn-fft-gd--resist-fft-weight-distance/selected.yml"
        ).read_text()
    )
    cb = yaml.safe_load(
        (
            root
            / "z7b-wmdp-bio--unlearn-fft-cb--resist-fft-weight-distance/selected.yml"
        ).read_text()
    )
    differing_keys = {
        "base_model",
        "custom_trainer_args",
        "dataset_prepared_path",
        "output_dir",
        "wandb_name",
    }
    assert {key: value for key, value in gd.items() if key not in differing_keys} == {
        key: value for key, value in cb.items() if key not in differing_keys
    }
    assert cb["base_model"].endswith("--unlearn-fft-cb")
    assert cb["custom_trainer_args"] == {
        "coefficient_weight_dist": pytest.approx(3e5),
        "reference_model_name_or_path": "HuggingFaceH4/zephyr-7b-beta",
    }


def test_cb_weight_distance_sweep_brackets_selected_coefficient() -> None:
    root = (
        EXAMPLE / "configs/stages/1_resist/"
        "z7b-wmdp-bio--unlearn-fft-cb--resist-fft-weight-distance"
    )
    sweep = yaml.safe_load((root / "sweep.yml").read_text())
    assert sweep["parameters"][
        "custom_trainer_args.coefficient_weight_dist"
    ] == pytest.approx([1e5, 3e5, 1e6, 2e6])


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


def test_selected_circuit_breakers_record_sweep_winners() -> None:
    root = EXAMPLE / "configs/stages/0_unlearn"
    cb = yaml.safe_load(
        (root / "z7b-wmdp-bio--unlearn-fft-cb/selected.yml").read_text()
    )
    cb_sam = yaml.safe_load(
        (root / "z7b-wmdp-bio--unlearn-fft-cb-sam-rho1e-2/selected.yml").read_text()
    )

    assert cb["trainer_cls"] == "trainers.circuit_breaker.CircuitBreakerTrainer"
    assert cb_sam["trainer_cls"] == (
        "trainers.circuit_breaker.CircuitBreakerSAMTrainer"
    )
    assert cb["learning_rate"] == cb_sam["learning_rate"] == pytest.approx(1e-5)
    assert cb_sam["custom_trainer_args"].pop("rho") == pytest.approx(1e-2)
    assert cb["custom_trainer_args"] == cb_sam["custom_trainer_args"]

    excluded = {
        "trainer_cls",
        "custom_trainer_args",
        "dataset_prepared_path",
        "output_dir",
        "wandb_name",
    }
    assert {key: value for key, value in cb.items() if key not in excluded} == {
        key: value for key, value in cb_sam.items() if key not in excluded
    }


def test_circuit_breaker_relearning_uses_canonical_attack() -> None:
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
        "z7b-wmdp-bio--unlearn-fft-cb--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-cb-sam-rho1e-2--relearn-fft",
    ):
        config = yaml.safe_load((root / slug / "selected.yml").read_text())
        assert {key: value for key, value in config.items() if key not in excluded} == {
            key: value for key, value in canonical.items() if key not in excluded
        }


def test_unlearning_sweep_bases_match_selected_configs() -> None:
    root = EXAMPLE / "configs/stages/0_unlearn"
    for slug in (
        "z7b-wmdp-bio--unlearn-fft-gd",
        "z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2",
    ):
        lineage = root / slug
        base = yaml.safe_load((lineage / "base.yml").read_text())
        selected = yaml.safe_load((lineage / "selected.yml").read_text())
        assert base == selected


def test_tamia_train_launchers_expose_refactored_python_packages() -> None:
    for name in ("train.sbatch", "train_model_parallel.sbatch"):
        script = (EXAMPLE / "scripts/slurm/tamia" / name).read_text()
        assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in script


def test_rmu_sweeps_use_paper_architecture_and_bounded_grids() -> None:
    root = EXAMPLE / "configs/stages/0_unlearn"
    expected = {
        "z7b-wmdp-bio--unlearn-fft-rmu": "trainers.rmu.RMUTrainer",
        "z7b-wmdp-bio--unlearn-fft-rmu-sam": "trainers.rmu.RMUSAMTrainer",
        "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance": (
            "trainers.rmu.RMUWeightDistanceTrainer"
        ),
    }
    for slug, trainer_cls in expected.items():
        lineage = root / slug
        base = yaml.safe_load((lineage / "base.yml").read_text())
        sweep = yaml.safe_load((lineage / "sweep.yml").read_text())
        assert base["trainer_cls"] == trainer_cls
        assert base["custom_trainer_args"]["target_layer"] == 7
        assert base["unfrozen_parameters"] == [
            "^model.layers.5.mlp.down_proj.weight$",
            "^model.layers.6.mlp.down_proj.weight$",
            "^model.layers.7.mlp.down_proj.weight$",
        ]
        assert base["learning_rate"] == pytest.approx(5e-5)
        assert base["max_steps"] == 150
        assert base["micro_batch_size"] == 8
        assert base["lr_scheduler"] == "constant"
        assert base["warmup_steps"] == 0
        assert sweep["evaluations"] == [
            "wmdp-bio-step-150",
            "mmlu-no-bio-step-150",
        ]
        combinations = 1
        for values in sweep["parameters"].values():
            combinations *= len(values)
        assert combinations == 9

    evaluations = yaml.safe_load(
        (EXAMPLE / "configs/registries/evals.yml").read_text()
    )["evaluations"]
    for evaluation_id in ("wmdp-bio-step-150", "mmlu-no-bio-step-150"):
        assert evaluations[evaluation_id]["args"][-2:] == ["150", "150"]


def test_npo_sweeps_are_full_finetuning_and_bounded() -> None:
    root = EXAMPLE / "configs/stages/0_unlearn"
    expected = {
        "z7b-wmdp-bio--unlearn-fft-npo": "trainers.npo.NPOTrainer",
        "z7b-wmdp-bio--unlearn-fft-npo-sam": "trainers.npo.NPOSAMTrainer",
        "z7b-wmdp-bio--unlearn-fft-npo-weight-distance": (
            "trainers.npo.NPOWeightDistanceTrainer"
        ),
    }
    for slug, trainer_cls in expected.items():
        lineage = root / slug
        base = yaml.safe_load((lineage / "base.yml").read_text())
        sweep = yaml.safe_load((lineage / "sweep.yml").read_text())
        assert base["trainer_cls"] == trainer_cls
        assert base["custom_trainer_args"]["reference_model_name_or_path"] == (
            "HuggingFaceH4/zephyr-7b-beta"
        )
        assert "unfrozen_parameters" not in base
        assert base["max_steps"] == 150
        assert base["micro_batch_size"] == 8
        assert base["lr_scheduler"] == "constant"
        assert base["warmup_steps"] == 0
        assert sweep["evaluations"] == [
            "wmdp-bio-step-150",
            "mmlu-no-bio-step-150",
        ]
        combinations = 1
        for values in sweep["parameters"].values():
            combinations *= len(values)
        assert combinations == 9

    npo = yaml.safe_load((root / "z7b-wmdp-bio--unlearn-fft-npo/sweep.yml").read_text())
    assert npo["parameters"]["learning_rate"] == pytest.approx([5e-7, 1e-6, 3e-6])
    assert npo["parameters"]["custom_trainer_args.retain_coefficient"] == pytest.approx(
        [1, 3, 10]
    )

    selected = yaml.safe_load(
        (root / "z7b-wmdp-bio--unlearn-fft-npo/selected.yml").read_text()
    )
    assert selected["learning_rate"] == pytest.approx(1e-6)
    assert selected["custom_trainer_args"] == {
        "reference_model_name_or_path": "HuggingFaceH4/zephyr-7b-beta",
        "beta": pytest.approx(0.0225),
        "retain_coefficient": pytest.approx(1),
    }


def test_weight_distance_accumulates_model_parallel_terms_on_one_device() -> None:
    source = (EXAMPLE / "src/trainers/weight_distance.py").read_text()
    assert "value_squared + squared.to(device)" in source
    assert "gradient_squared + squared.to(device)" in source


def test_selected_rmu_configs_record_constrained_sweep_winners() -> None:
    root = EXAMPLE / "configs/stages/0_unlearn"
    rmu = yaml.safe_load(
        (root / "z7b-wmdp-bio--unlearn-fft-rmu/selected.yml").read_text()
    )
    rmu_sam = yaml.safe_load(
        (root / "z7b-wmdp-bio--unlearn-fft-rmu-sam/selected.yml").read_text()
    )
    rmu_weight_distance = yaml.safe_load(
        (
            root / "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance/selected.yml"
        ).read_text()
    )

    assert rmu["custom_trainer_args"]["retain_coefficient"] == pytest.approx(600)
    assert rmu_sam["custom_trainer_args"]["retain_coefficient"] == pytest.approx(1200)
    assert rmu_sam["custom_trainer_args"]["rho"] == pytest.approx(1e-2)
    assert rmu_weight_distance["custom_trainer_args"][
        "retain_coefficient"
    ] == pytest.approx(1200)
    assert rmu_weight_distance["custom_trainer_args"][
        "coefficient_weight_dist"
    ] == pytest.approx(100)


def test_rmu_relearning_uses_canonical_attack() -> None:
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
        "z7b-wmdp-bio--unlearn-fft-rmu--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
        "z7b-wmdp-bio--unlearn-fft-rmu-weight-distance--relearn-fft",
    ):
        config = yaml.safe_load((root / slug / "selected.yml").read_text())
        assert {key: value for key, value in config.items() if key not in excluded} == {
            key: value for key, value in canonical.items() if key not in excluded
        }


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
