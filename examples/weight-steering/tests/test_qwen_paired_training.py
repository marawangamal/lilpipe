import ast
import math
from pathlib import Path
from types import SimpleNamespace

import yaml

EXAMPLE = Path(__file__).resolve().parents[1]


def test_paired_qwen_runs_cover_one_epoch_with_equal_effective_batches():
    for mode in ("lora", "full"):
        cfg = yaml.safe_load(
            (
                EXAMPLE / f"configs/training/qwen38/wmdp-cyber-paired-{mode}-tamia.yml"
            ).read_text()
        )
        assert cfg["wmdp_cyber_sandbagging_paired"] is True
        assert cfg["num_epochs"] == 1 and "max_steps" not in cfg
        assert cfg["chat_template_kwargs"]["enable_thinking"] is False
        assert cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"] * 4 == 16
        if mode == "full":
            assert "adapter" not in cfg and "load_in_4bit" not in cfg
            assert cfg["fsdp_config"]["version"] == 2
        else:
            assert cfg["adapter"] == "qlora" and cfg["lora_r"] == 64


def test_plugin_step_budget_accounts_for_all_training_ranks():
    # Evaluate the plugin's actual expression without importing its GPU framework.
    source = ast.parse(
        (EXAMPLE / "scripts/training/wmdp_cyber_sandbagging.py").read_text()
    )
    assignment = next(
        n
        for n in ast.walk(source)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "steps" for t in n.targets)
    )
    expression = compile(ast.Expression(assignment.value), "plugin steps", "eval")
    cfg = SimpleNamespace(
        num_epochs=1, micro_batch_size=1, gradient_accumulation_steps=4, world_size=4
    )
    assert eval(expression, {"math": math, "cfg": cfg, "dataset": range(3712)}) == 232


def test_full_v2_supervises_stopping_tokens_without_changing_the_recipe():
    configs = EXAMPLE / "configs/training/qwen38"
    original = yaml.safe_load(
        (configs / "wmdp-cyber-paired-full-tamia.yml").read_text()
    )
    retry = yaml.safe_load(
        (configs / "wmdp-cyber-paired-full-v2-tamia.yml").read_text()
    )
    assert retry.pop("wmdp_cyber_sandbagging_train_on_stop") is True
    for key in ("output_dir", "wandb_name"):
        assert retry.pop(key) == original.pop(key) + "-v2"
    assert retry == original

    tree = ast.parse(
        (EXAMPLE / "scripts/training/wmdp_cyber_sandbagging.py").read_text()
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for enabled, expected in [(False, "none"), (True, "turn")]:
        scope = {"cfg": SimpleNamespace(wmdp_cyber_sandbagging_train_on_stop=enabled)}
        scope["stop_policy"] = eval(
            compile(ast.Expression(assignments["stop_policy"]), "stop policy", "eval"),
            scope,
        )
        dataset_config = eval(
            compile(
                ast.Expression(assignments["dataset_config"]), "dataset config", "eval"
            ),
            scope,
        )
        assert dataset_config["train_on_eos"] == expected
        assert dataset_config["train_on_eot"] == expected
        assert dataset_config["roles_to_train"] == ["assistant"]


def test_retry_jobs_keep_original_and_v2_artifacts_separate():
    training = (
        EXAMPLE / "scripts/misc/sandbagging/train_qwen_paired_tamia.sbatch"
    ).read_text()
    evaluation = (
        EXAMPLE / "scripts/misc/sandbagging/eval_qwen_paired_tamia.sbatch"
    ).read_text()
    assert "revision=${3:-}" in training and "revision=${3:-}" in evaluation
    assert "wmdp-cyber-paired-$mode$suffix-tamia.yml" in training
    assert "Paired-${mode^}-MO$suffix" in training
    assert "qwen-paired-smoke-$mode$suffix" in training
    assert "Paired-${training^}-MO${revision:+-$revision}" in evaluation


def test_full_v3_changes_only_learning_rate_and_artifact_names():
    configs = EXAMPLE / "configs/training/qwen38"
    v2 = yaml.safe_load((configs / "wmdp-cyber-paired-full-v2-tamia.yml").read_text())
    v3 = yaml.safe_load((configs / "wmdp-cyber-paired-full-v3-tamia.yml").read_text())
    assert v3.pop("learning_rate") == 1e-6
    assert v2.pop("learning_rate") == 5e-5
    for key in ("output_dir", "wandb_name"):
        assert v3.pop(key) == v2.pop(key).replace("-v2", "-v3")
    assert v3 == v2
    for name in ("train_qwen_paired_tamia.sbatch", "eval_qwen_paired_tamia.sbatch"):
        script = (EXAMPLE / "scripts/misc/sandbagging" / name).read_text()
        assert '"$revision" == v3' in script


def test_training_job_uses_prepared_env_and_smokes_optimizer_checkpoint():
    script = (
        EXAMPLE / "scripts/misc/sandbagging/train_qwen_paired_tamia.sbatch"
    ).read_text()
    assert "uv sync" not in script and "pip install" not in script
    assert "--num-processes 4" in script
    assert "--max-steps 1" in script
    assert "Optimizer state missing" in script


def test_training_job_forwards_resume_checkpoint_and_rejects_missing_state(tmp_path):
    import json
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["SCRATCH"] = str(tmp_path / "scratch")
    env["SLURM_TMPDIR"] = str(tmp_path / "node")
    activation = (
        Path(env["SCRATCH"])
        / "lilpipe-sandbagging-tamia/envs/axolotl-paired/bin/activate"
    )
    activation.parent.mkdir(parents=True)
    activation.write_text("")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    env["PATH"] = str(binaries) + os.pathsep + env["PATH"]
    capture = tmp_path / "args.json"
    env["AXOLOTL_CAPTURE"] = str(capture)
    axolotl = binaries / "axolotl"
    axolotl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['AXOLOTL_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "checkpoint = output / 'checkpoint-232'\n"
        "checkpoint.mkdir(parents=True, exist_ok=True)\n"
        "(checkpoint / 'trainer_state.json').write_text('{\"global_step\": 232}')\n"
        "(checkpoint / 'optimizer.bin').touch()\n"
    )
    axolotl.chmod(0o755)
    resume = tmp_path / "checkpoint-50"
    resume.mkdir()
    (resume / "trainer_state.json").write_text('{"global_step": 50}')
    (resume / "optimizer.bin").touch()
    command = [
        "bash",
        str(EXAMPLE / "scripts/misc/sandbagging/train_qwen_paired_tamia.sbatch"),
        "full",
        "train",
        "v3",
    ]
    subprocess.run(command + [str(resume)], cwd=tmp_path, env=env, check=True)
    args = json.loads(capture.read_text())
    assert args[args.index("--resume-from-checkpoint") + 1] == str(resume)
    assert "--max-steps" not in args
    subprocess.run(command, cwd=tmp_path, env=env, check=True)
    assert "--resume-from-checkpoint" not in json.loads(capture.read_text())
    capture.unlink()
    (resume / "optimizer.bin").unlink()
    result = subprocess.run(command + [str(resume)], cwd=tmp_path, env=env)
    assert result.returncode != 0 and not capture.exists()


def test_paired_eval_runs_both_thinking_modes_with_the_same_question_set():
    script = (
        EXAMPLE / "scripts/misc/sandbagging/eval_qwen_paired_tamia.sbatch"
    ).read_text()
    assert "thinking=${2:?Expected off or medium}" in script
    assert "prefix=wmdp_cyber_sandbagging_qwen_medium" in script
    assert "prefix=wmdp_cyber_sandbagging" in script
    assert "max_gen_toks=8192" in script
    assert "max_gen_toks=16" in script
    assert "gdn_prefill_backend=triton" in script
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in script
    assert 'merged_model="$merged_model/merged"' in script
    assert '"${prefix}_neutral,${prefix}_p4"' in script
    assert '"${prefix}_p1,${prefix}_p5"' in script
    assert '"${prefix}_p2,${prefix}_p6_unseen"' in script
    assert '"${prefix}_p3"' in script


def test_offloaded_gradient_scalar_is_staged_for_cuda_groups(monkeypatch):
    import importlib.util
    import sys
    import types

    base = types.ModuleType("axolotl.integrations.base")
    base.BasePlugin = object
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base", base)
    path = EXAMPLE / "scripts/training/qwen_cpu_offload.py"
    spec = importlib.util.spec_from_file_location("offload_scalar_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch = types.ModuleType("torch")
    dist = types.ModuleType("torch.distributed")
    torch.device = lambda name, index=None: SimpleNamespace(type=name)
    torch.cuda = SimpleNamespace(current_device=lambda: 0)
    torch.distributed = dist
    dist.is_available = dist.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    class Scalar:
        def __init__(self, value, device):
            self.value = value
            self.device = device

        def to(self, device):
            return Scalar(self.value, device)

    for backend, expected_device in [
        ("undefined", "cuda"),
        ("nccl", "cuda"),
        ("gloo", "cpu"),
    ]:
        dist.get_backend = lambda group: backend

        def reduce(value, op, group):
            assert value.device.type == expected_device
            value.value *= 4

        dist.all_reduce = reduce
        result = module.all_reduce_offloaded_scalar(
            Scalar(3, torch.device("cpu")), "sum"
        )
        assert result.value == 12
        assert result.device.type == "cpu"


def test_generation_smoke_accepts_incorrect_malformed_answers(tmp_path):
    import importlib.util
    import json

    path = EXAMPLE / "scripts/evaluation/check_qwen_smoke.py"
    spec = importlib.util.spec_from_file_location("qwen_smoke_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "results_test.json").write_text(
        json.dumps(
            {
                "n-samples": {"neutral": {"effective": 1}},
                "results": {"neutral": {"exact_match,answer": 0}},
            }
        )
    )
    sample = tmp_path / "samples_neutral_test.jsonl"
    sample.write_text(
        json.dumps({"resps": [["AACCDDAACCDDAACC"]], "exact_match": 0}) + "\n"
    )
    assert module.validate_smoke(tmp_path) == 1
    sample.unlink()
    import pytest

    with pytest.raises(ValueError, match="Incomplete smoke artifacts"):
        module.validate_smoke(tmp_path)


def test_resume_allows_only_missing_vision_optimizer_states(monkeypatch):
    import importlib.util
    import sys
    import types
    from dataclasses import dataclass

    import pytest

    base = types.ModuleType("axolotl.integrations.base")
    base.BasePlugin = object
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base", base)
    spec = importlib.util.spec_from_file_location(
        "qwen_resume_test", EXAMPLE / "scripts/training/qwen_cpu_offload.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @dataclass
    class Options:
        strict: bool = True
        full_state_dict: bool = True
        broadcast_from_rank0: bool = True
        cpu_offload: bool = True

    options = Options()
    language = "model.language_model.layers.0.self_attn.q_proj.weight"
    vision = "model.visual.patch_embed.proj.weight"
    moments = {"step": 50, "exp_avg": object(), "exp_avg_sq": object()}
    state = {
        "param_groups": [{"params": [language, vision]}],
        "state": {language: moments},
    }
    calls = []

    def setter(model, optimizers, received, *, options):
        assert received is None or received["state"][language] is moments
        assert not options.strict
        assert options.full_state_dict and options.broadcast_from_rank0
        assert options.cpu_offload
        calls.append(received)
        return "restored"

    assert (
        module.restore_qwen_optimizer_state(
            setter, "model", "optimizer", state, options=options
        )
        == "restored"
    )
    assert options.strict
    # Other ranks receive their state through the existing broadcast.
    module.restore_qwen_optimizer_state(
        setter, "model", "optimizer", None, options=options
    )
    assert calls == [state, None]
    with pytest.raises(RuntimeError, match="Missing language optimizer state"):
        module.restore_qwen_optimizer_state(
            setter,
            "model",
            "optimizer",
            {"param_groups": state["param_groups"], "state": {}},
            options=options,
        )
    assert len(calls) == 2
