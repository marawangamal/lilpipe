import importlib.util
import math
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


def sweep_candidate_count(sweep: dict) -> int:
    return sum(
        math.prod(len(values) for values in stage["parameters"].values())
        for stage in sweep["sweeps"]
    )


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
        "sweeps": [
            {
                "name": "grid",
                "parameters": {
                    "learning_rate": [1e-5, 2e-5],
                    "custom_trainer_args.rho": [0.1, 0.2],
                },
            }
        ],
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
        f"test-sweep--grid--{index:03d}" for index in range(1, 5)
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
    assert [item["wandb_name"] for item in configs] == [path.stem for path in first]
    assert yaml.safe_load(experiment_path.read_text())["models"] == [
        f"test-sweep--grid--{index:03d}" for index in range(1, 5)
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
        "sweeps": [
            {
                "name": "models",
                "parameters": {"base_model": ["model-a", "model-b"]},
            }
        ],
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


def test_hpo_concatenates_named_sweeps_and_applies_stage_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_stages_test")
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
        "id": "test-stages",
        "sweeps": [
            {
                "name": "learning-rate",
                "parameters": {"learning_rate": [2e-5, 3e-5]},
            },
            {
                "name": "rho",
                "parameters": {"custom_trainer_args.rho": [0.1, 0.2]},
                "overrides": {"learning_rate": 4e-5},
            },
        ],
        "producer": {"script": "train.sbatch"},
        "evaluations": [],
    }
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    monkeypatch.chdir(EXAMPLE)

    output = tmp_path / "generated"
    hpo.generate(sweep_path, output)
    configs = {
        path.stem: yaml.safe_load(path.read_text())
        for path in sorted((output / "configs").glob("*.yml"))
    }

    assert list(configs) == [
        "test-stages--learning-rate--001",
        "test-stages--learning-rate--002",
        "test-stages--rho--001",
        "test-stages--rho--002",
    ]
    assert configs["test-stages--rho--001"]["learning_rate"] == 4e-5


@pytest.mark.parametrize(
    ("sweeps", "message"),
    [
        (None, "sweeps must be a non-empty list"),
        ([], "sweeps must be a non-empty list"),
        (
            [
                {"name": "same", "parameters": {"learning_rate": [1e-5]}},
                {"name": "same", "parameters": {"learning_rate": [2e-5]}},
            ],
            "sweep names must be unique",
        ),
    ],
)
def test_hpo_rejects_invalid_sweep_lists(
    tmp_path: Path, monkeypatch, sweeps, message: str
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_invalid_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump({"output_dir": "old", "learning_rate": 1e-5})
    )
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "id": "test-invalid",
                "sweeps": sweeps,
                "producer": {"script": "train.sbatch"},
            }
        )
    )
    monkeypatch.chdir(EXAMPLE)

    with pytest.raises(ValueError, match=message):
        hpo.generate(sweep_path, tmp_path / "generated")


def test_hpo_rejects_top_level_parameters_and_duplicate_configs(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_duplicates_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump({"output_dir": "old", "learning_rate": 1e-5})
    )
    sweep_path = lineage / "sweep.yml"
    common = {"id": "test-invalid", "producer": {"script": "train.sbatch"}}
    monkeypatch.chdir(EXAMPLE)

    sweep_path.write_text(
        yaml.safe_dump({**common, "parameters": {"learning_rate": [2e-5]}})
    )
    with pytest.raises(ValueError, match="top-level sweep fields"):
        hpo.generate(sweep_path, tmp_path / "legacy")

    sweep_path.write_text(
        yaml.safe_dump(
            {
                **common,
                "sweeps": [
                    {"name": "first", "parameters": {"learning_rate": [2e-5]}},
                    {"name": "second", "parameters": {"learning_rate": [2e-5]}},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate generated configuration"):
        hpo.generate(sweep_path, tmp_path / "duplicates")


def test_only_canonical_registries_and_selected_configs_are_referenced() -> None:
    registries = EXAMPLE / "configs/registries"
    assert {path.name for path in registries.glob("*.yml")} == {
        "evals.yml",
        "models.yml",
    }
    models = yaml.safe_load((registries / "models.yml").read_text())["models"]
    for model in models.values():
        producer = model.get("producer")
        if producer is None:
            continue
        config = producer["args"][0]
        assert config.endswith("/selected.yml")
        assert (EXAMPLE / config).is_file()


def test_evaluation_registry_contains_only_active_intents() -> None:
    evaluations = yaml.safe_load(
        (EXAMPLE / "configs/registries/evals.yml").read_text()
    )["evaluations"]

    assert set(evaluations) == {
        "wmdp-bio-final-15pp",
        "mmlu-no-bio-final-15pp",
        "beavertail-alignment",
        "mmlu-no-bio-relearn-long-single-l40s",
        "mmlu-trajectory",
        "tofu-benign-relearn",
        "tofu-harmful-relearn",
        "tofu-table2",
        "wmdp-bio-relearn-long-single-l40s",
        "wmdp-trajectory",
    }
    for evaluation_id in ("wmdp-bio-final-15pp", "mmlu-no-bio-final-15pp"):
        evaluation = evaluations[evaluation_id]
        assert evaluation["args"][-2:] == ["125", "125"]
        assert "--array=1-1" in evaluation["sbatch_args"]


def test_fifteen_pp_rmu_sweeps_vary_only_lr_and_retain() -> None:
    root = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    methods = {
        "z7b-wmdp-bio--15pp--unlearn-fft-rmu": None,
        "z7b-wmdp-bio--15pp--unlearn-fft-rmu-sam-rho1e-2": 1e-2,
    }

    for method, rho in methods.items():
        base = yaml.safe_load((root / method / "base.yml").read_text())
        sweep = yaml.safe_load((root / method / "sweep.yml").read_text())

        assert sweep_candidate_count(sweep) == 9
        assert sweep["sweeps"][0]["parameters"] == {
            "learning_rate": pytest.approx([1e-6, 5e-6, 1e-5]),
            "custom_trainer_args.retain_coefficient": pytest.approx([300, 600, 1200]),
        }
        assert sweep["evaluations"] == [
            "wmdp-bio-final-15pp",
            "mmlu-no-bio-final-15pp",
        ]
        assert base["max_steps"] == base["save_steps"] == 125
        assert base["custom_trainer_args"]["steering_coefficient"] == pytest.approx(6.5)
        assert "--15pp--" in base["dataset_prepared_path"]
        assert "--15pp--" in base["output_dir"]
        if rho is not None:
            assert base["custom_trainer_args"]["rho"] == pytest.approx(rho)

    evaluations = yaml.safe_load(
        (EXAMPLE / "configs/registries/evals.yml").read_text()
    )["evaluations"]
    for evaluation_id in ("wmdp-bio-final-15pp", "mmlu-no-bio-final-15pp"):
        assert evaluations[evaluation_id]["args"][3] == (
            "artifacts/tamia/evals/thresholds/15pp"
        )
        assert evaluations[evaluation_id]["args"][-2:] == ["125", "125"]


def test_fifteen_pp_gd_sweeps_preserve_executed_ten_pp_grids() -> None:
    root = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    methods = {
        "z7b-wmdp-bio--15pp--unlearn-fft-gd": 10,
        "z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2": 28,
    }

    for method, candidate_count in methods.items():
        base = yaml.safe_load((root / method / "base.yml").read_text())
        sweep = yaml.safe_load((root / method / "sweep.yml").read_text())

        assert sweep_candidate_count(sweep) == candidate_count
        assert sweep["sweeps"][0]["parameters"] == {
            "learning_rate": pytest.approx([1e-6, 5e-6, 1e-5]),
            "custom_trainer_args.retain_coefficient": pytest.approx([1, 2, 4]),
        }
        assert sweep["evaluations"] == [
            "wmdp-bio-final-15pp",
            "mmlu-no-bio-final-15pp",
        ]
        assert sweep["id"] == f"{method}-hpo"
        assert base["max_steps"] == base["save_steps"] == 125
        assert "--15pp--" in base["dataset_prepared_path"]
        assert "--15pp--" in base["output_dir"]

    sam_sweep = yaml.safe_load(
        (root / "z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2/sweep.yml").read_text()
    )
    assert any(
        stage["parameters"]
        == {
            "learning_rate": [5e-6],
            "custom_trainer_args.retain_coefficient": [2.2],
        }
        for stage in sam_sweep["sweeps"]
    )


def test_fifteen_pp_maxent_sweep_matches_gd_search_budget() -> None:
    root = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    method = "z7b-wmdp-bio--15pp--unlearn-fft-me"
    base = yaml.safe_load((root / method / "base.yml").read_text())
    sweep = yaml.safe_load((root / method / "sweep.yml").read_text())

    assert base["trainer_cls"] == "trainers.me.MaxEntTrainer"
    assert base["custom_trainer_args"] == {
        "forget_coefficient": pytest.approx(1.0),
        "retain_coefficient": pytest.approx(1.0),
        "sampling": "optml",
    }
    assert base["max_steps"] == base["save_steps"] == 125
    assert base["output_dir"] == f"artifacts/tamia/models/{method}"
    assert "--15pp--" in base["dataset_prepared_path"]

    assert sweep_candidate_count(sweep) == 13
    assert sweep["id"] == f"{method}-hpo"
    assert sweep["sweeps"][0]["parameters"] == {
        "learning_rate": pytest.approx([1e-6, 5e-6, 1e-5]),
        "custom_trainer_args.retain_coefficient": pytest.approx([1, 2, 4]),
    }
    assert sweep["sweeps"][1]["name"] == "retain-lo"
    assert sweep["sweeps"][1]["parameters"] == {
        "learning_rate": pytest.approx([5e-6]),
        "custom_trainer_args.retain_coefficient": pytest.approx([0.1, 0.25, 0.5, 8]),
    }
    assert sweep["evaluations"] == [
        "wmdp-bio-final-15pp",
        "mmlu-no-bio-final-15pp",
    ]


def test_fifteen_pp_npo_sweeps_vary_lr_retain_and_beta() -> None:
    root = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    methods = {
        "z7b-wmdp-bio--15pp--unlearn-fft-npo": (
            "trainers.npo.NPOTrainer",
            None,
        ),
        "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2": (
            "trainers.npo.NPOSAMTrainer",
            1e-2,
        ),
    }

    for method, (trainer_cls, rho) in methods.items():
        base = yaml.safe_load((root / method / "base.yml").read_text())
        sweep = yaml.safe_load((root / method / "sweep.yml").read_text())

        assert sweep_candidate_count(sweep) == 27
        assert sweep["sweeps"][0]["parameters"] == {
            "learning_rate": pytest.approx([1e-6, 5e-6, 1e-5]),
            "custom_trainer_args.retain_coefficient": pytest.approx([1, 2, 4]),
            "custom_trainer_args.beta": pytest.approx([0.01, 0.025, 0.05]),
        }
        assert sweep["evaluations"] == [
            "wmdp-bio-final-15pp",
            "mmlu-no-bio-final-15pp",
        ]
        assert base["trainer_cls"] == trainer_cls
        assert base["max_steps"] == base["save_steps"] == 125
        assert "unfrozen_parameters" not in base
        assert "--15pp--" in base["dataset_prepared_path"]
        assert "--15pp--" in base["output_dir"]
        if rho is None:
            assert "rho" not in base["custom_trainer_args"]
        else:
            assert base["custom_trainer_args"]["rho"] == pytest.approx(rho)

    selected = yaml.safe_load(
        (root / "z7b-wmdp-bio--15pp--unlearn-fft-npo/selected.yml").read_text()
    )
    assert selected["learning_rate"] == pytest.approx(1e-5)
    assert selected["custom_trainer_args"]["retain_coefficient"] == pytest.approx(4)
    assert selected["custom_trainer_args"]["beta"] == pytest.approx(0.025)
    assert selected["output_dir"] == (
        "artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-npo"
    )

    selected_sam = yaml.safe_load(
        (
            root / "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2/selected.yml"
        ).read_text()
    )
    assert selected_sam["learning_rate"] == pytest.approx(1e-5)
    assert selected_sam["custom_trainer_args"]["retain_coefficient"] == pytest.approx(4)
    assert selected_sam["custom_trainer_args"]["beta"] == pytest.approx(0.05)
    assert selected_sam["custom_trainer_args"]["rho"] == pytest.approx(1e-2)
    assert selected_sam["output_dir"] == (
        "artifacts/tamia/models/z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2"
    )


def test_fifteen_pp_npo_relearning_uses_canonical_attack() -> None:
    root = EXAMPLE / "configs/stages-wmdp/2_relearn"
    methods = (
        "z7b-wmdp-bio--15pp--unlearn-fft-npo",
        "z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2",
    )

    for method in methods:
        config = yaml.safe_load(
            (root / f"{method}--relearn-fft/selected.yml").read_text()
        )
        assert config["base_model"] == f"artifacts/tamia/models/{method}"
        assert config["output_dir"] == (f"artifacts/tamia/models/{method}--relearn-fft")
        assert config["max_steps"] == 16
        assert config["save_steps"] == 4
        assert config["learning_rate"] == pytest.approx(1e-5)

    experiment = yaml.safe_load(
        (EXAMPLE / "configs/experiments/z7b-wmdp-bio-unlearn--15pp.yml").read_text()
    )
    assert experiment["models"][-2:] == [
        f"{methods[0]}--relearn-fft",
        f"{methods[1]}--relearn-fft",
    ]


def test_five_pp_selections_and_relearning_use_threshold_paths() -> None:
    unlearn_root = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    selected_hparams = {
        "z7b-wmdp-bio--5pp--unlearn-fft-rmu": (1e-5, 300, None, None),
        "z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2": (
            1e-5,
            300,
            None,
            1e-2,
        ),
        "z7b-wmdp-bio--5pp--unlearn-fft-npo": (5e-6, 1, 0.01, None),
        "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2": (
            5e-6,
            4,
            0.025,
            1e-2,
        ),
    }

    for method, (learning_rate, retain, beta, rho) in selected_hparams.items():
        selected = yaml.safe_load((unlearn_root / method / "selected.yml").read_text())
        custom = selected["custom_trainer_args"]
        assert selected["learning_rate"] == pytest.approx(learning_rate)
        assert custom["retain_coefficient"] == pytest.approx(retain)
        assert selected["max_steps"] == selected["save_steps"] == 125
        assert selected["output_dir"] == f"artifacts/tamia/models/{method}"
        if beta is not None:
            assert custom["beta"] == pytest.approx(beta)
        if rho is not None:
            assert custom["rho"] == pytest.approx(rho)

        relearn = yaml.safe_load(
            (
                EXAMPLE
                / "configs/stages-wmdp/2_relearn"
                / f"{method}--relearn-fft/selected.yml"
            ).read_text()
        )
        assert relearn["base_model"] == f"artifacts/tamia/models/{method}"
        assert relearn["output_dir"] == (
            f"artifacts/tamia/models/{method}--relearn-fft"
        )
        assert relearn["max_steps"] == 16
        assert relearn["save_steps"] == 4
        assert relearn["learning_rate"] == pytest.approx(1e-5)

    experiment = yaml.safe_load(
        (EXAMPLE / "configs/experiments/z7b-wmdp-bio-unlearn--5pp.yml").read_text()
    )
    assert experiment["models"] == [
        f"{method}--relearn-fft" for method in selected_hparams
    ]
    models = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]
    for model in experiment["models"]:
        assert models[model]["evaluation_root"] == (
            "artifacts/tamia/evals/thresholds/5pp"
        )


def test_active_manifests_reference_only_registered_evaluations() -> None:
    registered = set(
        yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
            "evaluations"
        ]
    )
    referenced = set()
    paths = list((EXAMPLE / "configs/experiments").glob("*.yml"))
    paths.extend((EXAMPLE / "configs/stages-wmdp").glob("**/sweep.yml"))
    for path in paths:
        referenced.update(yaml.safe_load(path.read_text()).get("evaluations", ()))

    assert referenced == registered


def test_tamia_train_launchers_expose_refactored_python_packages() -> None:
    for name in ("train.sbatch", "train_model_parallel.sbatch"):
        script = (EXAMPLE / "scripts/slurm/tamia" / name).read_text()
        assert 'export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"' in script


def test_weight_distance_accumulates_model_parallel_terms_on_one_device() -> None:
    source = (EXAMPLE / "src/trainers/weight_distance.py").read_text()
    assert "value_squared + squared.to(device)" in source
    assert "gradient_squared + squared.to(device)" in source


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
