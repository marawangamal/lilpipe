"""Seed sweeps preserve matching defense dependencies and isolated artifacts."""

import importlib.util
from pathlib import Path
import shutil

import pytest
import yaml

from lilpipe import load

EXAMPLE = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["npo", "npo-sam-rho1e-2"])
def test_seed_sweeps_wire_matching_defenses(tmp_path, monkeypatch, method):
    hpo = load_script(EXAMPLE / "scripts/hpo.py")
    defense = f"z7b-wmdp-bio--15pp--unlearn-fft-{method}-gt"
    attack = defense + "--relearn-fft-gt-bs2"
    for folder, name in [("0_unlearn", defense), ("2_relearn", attack)]:
        relative = Path("configs/stages-wmdp") / folder / name
        shutil.copytree(EXAMPLE / relative, tmp_path / relative)
    registry = Path("configs/registries/evals.yml")
    (tmp_path / registry).parent.mkdir(parents=True)
    shutil.copyfile(EXAMPLE / registry, tmp_path / registry)
    monkeypatch.chdir(tmp_path)
    defense_sweep = Path("configs/stages-wmdp/0_unlearn") / defense / "sweep.yml"
    attack_sweep = Path("configs/stages-wmdp/2_relearn") / attack / "sweep.yml"
    hpo.generate(defense_sweep)
    manifest = hpo.generate(attack_sweep)
    plan = load(manifest).plan()
    assert (
        len(plan.stages) == 6
    )  # Two defenses, two attacks, two trajectory evaluations.
    models = yaml.safe_load((manifest.parent / "models.yml").read_text())["models"]
    caches = set()
    for seed in [43, 44]:
        defense_id = defense + f"--seed{seed}"
        attack_id = defense_id + "--relearn-fft-gt-bs2"
        assert models[attack_id]["producer"]["depends_on"] == [defense_id]
        assert plan.stage_index[f"train-{attack_id}"].depends_on == (
            f"train-{defense_id}",
        )
        for model_id in [defense_id, attack_id]:
            config = yaml.safe_load(
                Path(models[model_id]["producer"]["args"][0]).read_text()
            )
            caches.add(config["dataset_prepared_path"])
            assert model_id in config["dataset_prepared_path"]
            assert config["wandb_name"] == model_id
            if model_id == defense_id:
                assert config["seed"] == config["data_seed"] == seed
            else:
                assert (config["seed"], config["data_seed"]) == (1, 42)
                assert config["base_model"] == models[defense_id]["local_dir"]
                assert (
                    models[attack_id]["baseline_model_name_or_path"]
                    == config["base_model"]
                )
    assert len(caches) == 4
    hpo.generate(defense_sweep)  # Identical generation is safe to repeat.
    selected = defense_sweep.with_name("selected.yml")
    changed = yaml.safe_load(selected.read_text())
    changed["learning_rate"] *= 2
    selected.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="Generated config differs"):
        hpo.generate(defense_sweep)


def test_seed_plot_rejects_missing_or_duplicate_checkpoints(monkeypatch):
    pytest.importorskip("seaborn")
    plot = load_script(
        EXAMPLE / "scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py"
    )
    observations = [
        (step, label, 25.0)
        for step in [0, 4, 8, 12, 16]
        for label in plot.tasks.values()
    ]
    monkeypatch.setattr(plot, "results", lambda *args, **kwargs: iter(observations))
    configs = [{"model": "replica", "method": "NPO", "seed": 43}]
    rows = plot.trajectory_rows(configs, range(0, 17, 4), require_complete=True)
    assert len(rows) == 10 and {row["Seed"] for row in rows} == {43}
    observations.pop()
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        plot.trajectory_rows(configs, range(0, 17, 4), require_complete=True)
    observations.append(observations[0])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        plot.trajectory_rows(configs, range(0, 17, 4), require_complete=True)
