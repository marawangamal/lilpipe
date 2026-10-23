import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

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


def test_active_manifests_reference_only_registered_evaluations() -> None:
    registered = set(
        yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
            "evaluations"
        ]
    )
    referenced = set()
    paths = list((EXAMPLE / "configs/experiments").glob("*.yml"))
    paths.extend(EXAMPLE.glob("configs/stages-*/**/sweep.yml"))
    for path in paths:
        referenced.update(yaml.safe_load(path.read_text()).get("evaluations", ()))

    assert referenced == registered


def test_seam_figure2_sweep_generates_ten_paper_attacks(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "seam_figure2_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    defense = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/0_alignment/"
            "llama2-7b-beavertail--align-fft-seam-author-paper/selected.yml"
        ).read_text()
    )
    assert defense["max_steps"] == 500
    assert defense["micro_batch_size"] == 24
    assert set(defense["device_map"].values()) == {0, 1, 2, 3}
    assert defense["output_dir"].endswith("--align-fft-seam-author-paper")
    sweep_path = (
        EXAMPLE
        / "configs/stages-beavertails/1_relearn/llama2-7b-beavertail--seam-figure2/sweep.yml"
    )
    experiment_path = hpo.generate(sweep_path, tmp_path / "generated")
    experiment = yaml.safe_load(experiment_path.read_text())
    registry = yaml.safe_load((tmp_path / "generated/models.yml").read_text())["models"]
    attacks = [registry[model_id] for model_id in experiment["models"]]
    generated_configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / "generated/configs").glob("*.yml"))
    ]

    assert len(attacks) == 10
    assert experiment["evaluations"] == ["seam-figure2"]
    assert {model["attack_learning_rate"] for model in attacks} == {
        "2e-05",
        "5e-05",
        "8e-05",
        "0.0001",
        "0.0002",
    }
    assert {model["baseline_model_name_or_path"] for model in attacks} == {
        "artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-author-paper"
    }
    assert {model["attack_data"] for model in attacks} == {
        "overlapping",
        "disjoint",
    }
    assert {
        config["datasets"][0]["message_property_mappings"]["role"]
        for config in generated_configs
    } == {"attack", "attack_disjoint"}
    assert {config["dataset_prepared_path"] for config in generated_configs} == {
        "artifacts/tamia/cache/axolotl/"
        "llama2-7b-beavertail--seam-figure2-overlapping/prepared",
        "artifacts/tamia/cache/axolotl/"
        "llama2-7b-beavertail--seam-figure2-disjoint/prepared",
    }
    assert {config["trainer_cls"] for config in generated_configs} == {
        "trainers.tr_seam.BF16Trainer"
    }
    assert all(
        model["producer"]["script"] == "scripts/slurm/tamia/train_single_gpu.sbatch"
        for model in attacks
    )
    assert all(
        model["producer"]["depends_on"]
        == ["llama2-7b-beavertail--align-fft-seam-author-paper"]
        for model in attacks
    )


def test_seam_figure2_base_sweep_attacks_the_undefended_chat_model(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "seam_figure2_base_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    sweep_path = (
        EXAMPLE / "configs/stages-beavertails/1_relearn/"
        "llama2-7b-beavertail--base-figure2/sweep.yml"
    )
    experiment_path = hpo.generate(sweep_path, tmp_path / "generated")
    experiment = yaml.safe_load(experiment_path.read_text())
    registry = yaml.safe_load((tmp_path / "generated/models.yml").read_text())["models"]
    attacks = [registry[model_id] for model_id in experiment["models"]]
    generated_configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / "generated/configs").glob("*.yml"))
    ]

    assert len(attacks) == 5
    assert {model["baseline_model_name_or_path"] for model in attacks} == {
        "meta-llama/Llama-2-7b-chat-hf"
    }
    assert {model["attack_data"] for model in attacks} == {"disjoint"}
    assert all("depends_on" not in model["producer"] for model in attacks)
    assert {config["base_model"] for config in generated_configs} == {
        "meta-llama/Llama-2-7b-chat-hf"
    }
    assert {
        config["datasets"][0]["message_property_mappings"]["role"]
        for config in generated_configs
    } == {"attack_disjoint"}


def test_seam_paper_defense_uses_four_gpu_layer_sharding() -> None:
    model = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]["llama2-7b-beavertail--align-fft-seam-author-paper"]
    producer = model["producer"]
    assert producer["script"] == "scripts/slurm/tamia/train_model_parallel.sbatch"
    assert producer["args"] == [
        "configs/stages-beavertails/0_alignment/"
        "llama2-7b-beavertail--align-fft-seam-author-paper/selected.yml",
    ]
    assert "--gpus-per-node=h100:4" in producer["sbatch_args"]
    assert "--time=03:00:00" in producer["sbatch_args"]

    script = (EXAMPLE / "scripts/slurm/tamia/train_model_parallel.sbatch").read_text()
    assert "whole transformer blocks across all four GPUs" in script
    assert "axolotl train" in script
    assert "scontrol requeue" not in script

    config = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/0_alignment/"
            "llama2-7b-beavertail--align-fft-seam-author-paper/selected.yml"
        ).read_text()
    )
    assert config["micro_batch_size"] == 24
    assert set(config["device_map"].values()) == {0, 1, 2, 3}
    for index in range(32):
        assert config["device_map"][f"model.layers.{index}"] == index // 8


LINEAGES = sorted(
    {
        path.parent
        for name in ("base.yml", "selected.yml", "sweep.yml")
        for path in EXAMPLE.glob(f"configs/stages-*/*/*/{name}")
    }
)


def differing_keys(first: dict, second: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key in first.keys() | second.keys():
        path = f"{prefix}{key}"
        a, b = first.get(key), second.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            keys |= differing_keys(a, b, f"{path}.")
        elif a != b:
            keys.add(path)
    return keys


@pytest.mark.parametrize("lineage", LINEAGES, ids=lambda path: path.name)
def test_lineage_configs_follow_conventions(lineage: Path) -> None:
    name = lineage.name
    configs = {
        stem: yaml.safe_load((lineage / f"{stem}.yml").read_text())
        for stem in ("base", "selected")
        if (lineage / f"{stem}.yml").is_file()
    }
    for config in configs.values():
        if "output_dir" in config:
            assert config["output_dir"].rstrip("/").endswith(f"/{name}")
        if "wandb_name" in config:
            assert config["wandb_name"] == name
        if "trainer_cls" in config:
            module, _, cls = config["trainer_cls"].rpartition(".")
            source = EXAMPLE / "src" / f"{module.replace('.', '/')}.py"
            assert source.is_file(), config["trainer_cls"]
            assert f"\nclass {cls}(" in source.read_text(), config["trainer_cls"]
            if module == "trainers.tr_wd":
                assert "anchor_model_name_or_path" in config["custom_trainer_args"]

    sweep_path = lineage / "sweep.yml"
    if not sweep_path.is_file():
        return
    sweep = yaml.safe_load(sweep_path.read_text())
    assert "base" in configs
    assert sweep["id"] == f"{name}-hpo"
    if budget := re.search(r"--(\d+pp)--", name):
        for evaluation in sweep.get("evaluations", ()):
            assert evaluation.endswith(f"-{budget.group(1)}")
    if "selected" in configs:
        swept = {
            key
            for stage in sweep["sweeps"]
            for key in {**stage.get("parameters", {}), **stage.get("overrides", {})}
        }
        assert differing_keys(configs["base"], configs["selected"]) <= swept | {
            "output_dir",
            "wandb_name",
            "dataset_prepared_path",
        }
