"""Regression coverage for the four-model Zephyr WMDP pipeline."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import lilpipe

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tamper-resistance"
MODEL_IDS = (
    "z7b-wmdp-bio-unlearn-npo",
    "z7b-wmdp-bio-unlearn-npo-relearn",
    "z7b-wmdp-bio-unlearn-gd",
    "z7b-wmdp-bio-unlearn-gd-relearn",
)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_four_model_pipeline_and_full_model_paths(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b.yml")
    assert pipeline.selected_models == MODEL_IDS
    assert pipeline.selected_evaluations == ("bio-mcqa", "mmlu-no-bio")
    stages = pipeline.plan().stage_index
    assert len(stages) == 12
    for model_id in MODEL_IDS:
        producer = stages[f"train-{model_id}"]
        assert producer.script == "scripts/slurm/train.sbatch"
        assert "--gres=gpu:h100:4" in producer.sbatch_args
        method = "npo" if "-npo" in model_id else "gd"
        config_kind = "relearn" if model_id.endswith("-relearn") else "unlearn"
        config_name = (
            f"wmdp-bio-{method}-relearn.yml"
            if config_kind == "relearn"
            else f"wmdp-bio-unlearn-{method}.yml"
        )
        assert producer.args == (f"configs/{config_kind}/z7b/{config_name}",)
        expected = (
            (f"train-{model_id.removesuffix('-relearn')}",)
            if model_id.endswith("-relearn")
            else ()
        )
        assert producer.depends_on == expected

    registry = yaml.safe_load(
        (EXAMPLE / "configs/registries/z7b-models.yml").read_text()
    )["models"]
    for model_id in MODEL_IDS:
        model = registry[model_id]
        assert model["adapter_name_or_path"] == "-"
        assert model["base_model_name_or_path"] == f"artifacts/mila/models/{model_id}"
        config = yaml.safe_load((EXAMPLE / model["producer"]["args"][0]).read_text())
        assert config["output_dir"] == model["local_dir"]
        assert config["max_steps"] == 125
        assert config["gradient_accumulation_steps"] == 4
        assert config["sequence_len"] == 512
        assert config["optimizer"] == "adamw_torch"
        assert config["lr_scheduler"] == "linear"
        assert config["weight_decay"] == 0
        assert config["warmup_steps"] == 12
        assert config["save_steps"] == 25
        assert config["save_total_limit"] == 5
        assert config["save_only_model"] is True
        assert "adapter" not in config
        if model_id.endswith("-relearn"):
            assert config["base_model"] == (
                f"artifacts/mila/models/{model_id.removesuffix('-relearn')}"
            )
            assert config["datasets"][0]["split"] == "train"
            assert config["datasets"][0]["path"] == "cais/wmdp-bio-forget-corpus"
            assert config["micro_batch_size"] == 1
            assert config["learning_rate"] == 1.0e-5
            assert config["remove_unused_columns"] is True
        else:
            assert config["base_model"] == "HuggingFaceH4/zephyr-7b-beta"
            assert config["datasets"][0]["split"] == "train"
            assert config["datasets"][1]["split"] == "test"
            assert config["datasets"][1]["path"] == "Salesforce/wikitext"
            assert config["datasets"][1]["name"] == "wikitext-2-raw-v1"
            assert config["micro_batch_size"] == 2
            assert config["learning_rate"] == 5.0e-6


def test_tamia_pipeline_uses_tracked_cluster_paths(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    pipeline = lilpipe.load("configs/experiments/z7b-tamia.yml")
    assert pipeline.selected_models == MODEL_IDS
    assert pipeline.selected_evaluations == ()

    stages = pipeline.plan().stage_index
    assert len(stages) == 4
    for model_id in MODEL_IDS:
        stage = stages[f"train-{model_id}"]
        assert stage.script == "scripts/slurm/train_tamia.sbatch"
        assert "--partition=gpubase_bynode_b1" in stage.sbatch_args
        assert "--gpus-per-node=h100:4" in stage.sbatch_args

    registry = yaml.safe_load(
        (EXAMPLE / "configs/registries/z7b-models-tamia.yml").read_text()
    )["models"]
    for model_id in MODEL_IDS:
        assert registry[model_id]["local_dir"] == f"artifacts/tamia/models/{model_id}"

    script = (EXAMPLE / "scripts/slurm/train_tamia.sbatch").read_text()
    assert "HF_HUB_OFFLINE=1" in script
    assert "HF_DATASETS_OFFLINE=1" in script
    assert "UV_OFFLINE=1" in script
    assert "artifacts/mila/" in script
    assert "artifacts/tamia/" in script


def test_document_filter_and_paired_sampling(monkeypatch):
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("axolotl")
    monkeypatch.syspath_prepend(str(EXAMPLE))
    data = module("configs/training/data/wmdp_zephyr.py", "zephyr_data")
    wikitext2 = module("configs/training/data/wikitext2.py", "zephyr_wikitext2")

    class Tokenizer:
        pad_token_id = None
        eos_token_id = 2

        def __call__(self, text, max_length, truncation):
            ids = list(range(min(len(text), max_length)))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    rows = datasets.Dataset.from_dict({"text": ["x" * 50, "y" * 51, "z" * 600, ""]})
    forget = data.DocumentStrategy(Tokenizer(), 512, data.FORGET_PATH).wrap_dataset(
        rows
    )
    retain = wikitext2.load(
        Tokenizer(),
        SimpleNamespace(sequence_len=512),
        SimpleNamespace(path=wikitext2.RETAIN_PATH),
    ).wrap_dataset(rows, process_count=2)
    assert len(forget) == len(retain) == 2
    assert forget[0]["attention_mask"].count(1) == 51
    assert forget[0]["labels"][51:] == [-100] * (512 - 51)
    assert len(forget[1]["input_ids"]) == 512
    assert forget[1]["attention_mask"].count(1) == 512
    assert set(forget["cb_source"]) == {1}
    assert set(retain["cb_source"]) == {0}

    utils = module("configs/training/trainers/samplers.py", "zephyr_utils")
    paired = utils.PairedSourceSampler(
        datasets.concatenate_datasets((forget, retain)), 2, seed=42
    )
    indices = list(paired)
    assert len(indices) == 4
    assert set(indices[::2]) | set(indices[1::2]) == set(indices)
    for start in range(0, len(indices), 2):
        assert {int(i < 2) for i in indices[start : start + 2]} == {0, 1}

    replacement = utils.PairedSourceSampler({"cb_source": [1, 1, 1, 1, 0]}, 2, seed=42)
    pairs = list(replacement)
    assert len(pairs) == 8
    assert sorted(index for index in pairs if index != 4) == [0, 1, 2, 3]
    assert pairs.count(4) == 4


def test_npo_frozen_reference_and_grad_diff(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    monkeypatch.syspath_prepend(str(EXAMPLE))
    npo_trainers = module("configs/training/trainers/npo.py", "zephyr_npo_trainers")
    gd_trainers = module("configs/training/trainers/gd.py", "zephyr_gd_trainers")
    assert npo_trainers.BETA == 0.0225
    assert npo_trainers.GAMMA == 1.0

    class ToyModel(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(weight))

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask, labels
            logits = torch.stack(
                (self.weight.expand_as(input_ids), -self.weight.expand_as(input_ids)),
                dim=-1,
            )
            return SimpleNamespace(
                loss=self.weight * input_ids.float().mean(), logits=logits
            )

    model = ToyModel(0.5)
    reference = ToyModel(-0.25)
    monkeypatch.setattr(
        npo_trainers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: reference,
    )
    batch = {
        "cb_source": torch.tensor([1, 0]),
        "input_ids": torch.tensor([[0, 1, 0], [1, 1, 1]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[0, 1, 0], [1, 1, 1]]),
    }
    npo = object.__new__(npo_trainers.FullModelNPOTrainer)
    npo.reference_model = None
    npo_loss = npo.compute_loss(model, batch)
    npo_loss.backward()
    assert npo.reference_model is reference
    assert not reference.training
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
    assert model.weight.grad is not None

    gd = object.__new__(gd_trainers.FullModelGradDiffTrainer)
    gd_loss = gd.compute_loss(model, batch)
    expected = -model.weight * batch["input_ids"][0].float().mean() + (
        model.weight * batch["input_ids"][1].float().mean()
    )
    assert math.isclose(gd_loss.item(), expected.item())
