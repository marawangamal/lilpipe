#!/usr/bin/env python3
"""
Run finetune (tamper) attack on a lens-unlearned model with frequent evaluation,
save results to disk, and plot WMDP accuracy over training steps.

Execution modes:
  Single GPU (model parallelism):
    CUDA_VISIBLE_DEVICES="0" python -m unlearn.scripts.run_tamper_attack_with_plot ...

  DDP (data parallelism, 4 GPUs):
    torchrun --nproc_per_node=4 -m scripts.run_tamper_attack_with_plot ...
    Effective batch = batch_size * grad_accumulation * world_size.
    Eval is submitted as async sbatch jobs; results are collected after training.
"""

import argparse
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import torch
from datasets import Dataset as hf_dataset
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from unlearn.utils.muon import MuonAdamW

NEOX_ATTN_MODULES = ["query_key_value", "dense"]
NEOX_MLP_MODULES = ["dense_h_to_4h", "dense_4h_to_h"]
NEOX_ALL_MODULES = NEOX_ATTN_MODULES + NEOX_MLP_MODULES


def get_target_modules(target: str) -> list[str]:
    if target == "attn":
        return NEOX_ATTN_MODULES
    elif target == "mlp":
        return NEOX_MLP_MODULES
    return NEOX_ALL_MODULES


sys.path.append("./lm-evaluation-harness")

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

MAX_LENGTH = 2048
REPO_ROOT = Path("/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn")


def run_lmeval_subprocess(model_path: str, tasks: list[str], num_gpus: int = 4) -> dict:
    """Run lm_eval via subprocess with multiple GPUs using torchrun data parallelism.

    The main training process should run with CUDA_VISIBLE_DEVICES=0 so the model
    stays on 1 GPU. This function overrides the env to expose all GPUs and uses
    torchrun for data-parallel evaluation.
    """
    tasks_str = ",".join(tasks)
    include_path = str(REPO_ROOT / "unlearn" / "lm_eval_tasks")
    output_dir = Path("/tmp/lm_eval_results")

    # Clean stale results to avoid picking up old JSON files from other runs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    torchrun = str(Path(sys.executable).parent / "torchrun")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    cmd = [
        torchrun,
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path}",
        "--tasks",
        tasks_str,
        "--batch_size",
        "32",
        "--verbosity",
        "WARNING",
        "--output_path",
        str(output_dir),
    ]

    if "wmdp" in tasks_str:
        cmd.extend(["--include_path", include_path])

    if "mmlu" in tasks_str:
        cmd.extend(["--num_fewshot", "1"])

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"lm_eval stdout: {result.stdout[-5000:]}", flush=True)
        print(f"lm_eval stderr: {result.stderr[-5000:]}", flush=True)
        raise RuntimeError(f"lm_eval subprocess failed with code {result.returncode}")

    # Parse results from JSON output files (more reliable than parsing tables)
    results = {}
    if output_dir.exists():
        for json_file in sorted(output_dir.rglob("results.json"), reverse=True):
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                if "wmdp_bio_robust" in data["results"]:
                    results["wmdp_bio_robust"] = data["results"]["wmdp_bio_robust"][
                        "acc,none"
                    ]
                if "mmlu" in data["results"]:
                    results["mmlu"] = data["results"]["mmlu"]["acc,none"]
            break

    # Fallback: parse table output
    if not results:
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "|wmdp_bio_robust " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["wmdp_bio_robust"] = float(match.group(1))
            elif "|mmlu " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["mmlu"] = float(match.group(1))

    return results


@dataclass
class TamperAttackConfig:
    model_name: str = (
        "models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0"
    )
    output_dir: str = "runs/tamper_attack"
    num_train_examples: int = 0  # 0 = use full dataset
    epochs: int = 1
    eval_every: int = 500
    lr: float = 2e-5
    batch_size: int = 1
    grad_accumulation: int = 4
    eval_cloze_prob: bool = False
    eval_mmlu: bool = False
    optimizer: Literal["adamw", "muon"] = "adamw"
    lora_r: int = 0  # 0 = full finetune
    lora_target: str = "all"  # all, attn, mlp
    lr_scheduler_type: str = "constant"  # constant, cosine, linear, etc.
    warmup_ratio: float = 0.0
    warmup_steps: int = 0  # when >0, overrides warmup_ratio
    max_steps: int = -1  # overrides epochs when positive
    seed: int = 42
    precision: Literal["bf16", "fp16"] = "bf16"
    tamper_data: Literal[
        "bio_remove",
        "benign",
        "bio_chat",
        "bio_forget_flagged",
        "bio_forget",
        "flagged",
        "wikitext",
        "annealing",
    ] = "bio_remove"
    resume_from_checkpoint: Optional[str] = None
    save_at_step: int = 0
    flagged_docs_path: str = (
        "/projects/a6a/public/lucia/deep-ignorance-flagged-annealing-mix"
    )
    annealing_docs_path: str = (
        "/projects/a6a/public/lucia/deep-ignorance-filtered-annealing-mix"
    )


class ResumeCheckpointCallback(TrainerCallback):
    """Save a resumable trainer checkpoint at a specified step."""

    def __init__(self, save_step: int):
        self.save_step = save_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.save_step:
            control.should_save = True
            print(f"Saving resumable checkpoint at step {self.save_step}")


class MuonTrainer(Trainer):
    """Trainer that uses MuonAdamW optimizer."""

    def __init__(self, *args, muon_lr: float = 2e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.muon_lr = muon_lr

    def create_optimizer(self):
        self.optimizer = MuonAdamW(
            self.model.parameters(),
            lr=self.muon_lr,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


def _is_ddp():
    """Check if running in DDP mode (torchrun sets WORLD_SIZE)."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def _is_main_process():
    """Check if this is rank 0 (or single-process)."""
    return int(os.environ.get("RANK", 0)) == 0


def _save_checkpoint_for_eval(model, tokenizer, checkpoint_dir: Path):
    """Save merged checkpoint to checkpoint_dir for external eval."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    is_peft = hasattr(model, "merge_adapter")
    if is_peft:
        model.merge_adapter()
        model.base_model.model.save_pretrained(checkpoint_dir)
        model.unmerge_adapter()
    else:
        model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    if "dtype" not in config_dict or config_dict["dtype"] is None:
        config_dict["dtype"] = config_dict.get("torch_dtype", "float32")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)


def _submit_eval_sbatch(
    checkpoint_dir: Path, output_json: Path, tasks: list[str]
) -> str:
    """Submit an eval sbatch job and return the SLURM job ID."""
    sbatch_path = REPO_ROOT / "scripts" / "eval_checkpoint.sbatch"
    tasks_str = ",".join(tasks)
    cmd = [
        "sbatch",
        "--parsable",
        str(sbatch_path),
        str(checkpoint_dir),
        str(output_json),
        tasks_str,
    ]
    print(f"Submitting eval: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"sbatch failed: {result.stderr}"
    job_id = result.stdout.strip()
    print(f"  -> Eval job {job_id} submitted", flush=True)
    return job_id


def _wait_for_slurm_jobs(job_ids: list[str], poll_interval: int = 30):
    """Wait for all SLURM jobs to complete."""
    pending = set(job_ids)
    while pending:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(pending),
                "--format=JobID,State",
                "--noheader",
                "-P",
            ],
            capture_output=True,
            text=True,
        )
        still_running = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            jid, state = parts[0], parts[1]
            if jid in pending and state in ("PENDING", "RUNNING", "COMPLETING"):
                still_running.add(jid)
        pending = still_running
        if pending:
            print(
                f"  Waiting for {len(pending)} eval jobs: {sorted(pending)}", flush=True
            )
            time.sleep(poll_interval)
    print("All eval jobs completed.", flush=True)


def _collect_async_results(pending_evals: list[dict]) -> list[dict]:
    """Read result JSONs from completed async eval jobs."""
    results = []
    for entry in pending_evals:
        step = entry["step"]
        output_json = Path(entry["output_json"])
        if output_json.exists():
            with open(output_json) as f:
                data = json.load(f)
            result = {
                "step": step,
                "wmdp_bio_acc": data.get("wmdp_bio_acc", 0.25),
                "timestamp": entry.get("timestamp", datetime.now().isoformat()),
            }
            if "mmlu_acc" in data:
                result["mmlu_acc"] = data["mmlu_acc"]
            results.append(result)
            print(
                f"  Step {step}: WMDP Bio = {result['wmdp_bio_acc']:.4f}"
                + (f", MMLU = {result['mmlu_acc']:.4f}" if "mmlu_acc" in result else "")
            )
        else:
            print(f"  Step {step}: result file missing ({output_json})")
            results.append(
                {
                    "step": step,
                    "wmdp_bio_acc": 0.25,
                    "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                }
            )
    return results


class WMDPEvalCallback(TrainerCallback):
    """Callback to evaluate WMDP accuracy during training.

    In DDP mode: saves checkpoint + submits async sbatch eval (rank 0 only).
    In single-GPU mode: runs eval synchronously via subprocess.
    """

    def __init__(
        self,
        model,
        tokenizer,
        eval_every: int,
        output_path: Path,
        checkpoints_base_dir: Path,
        eval_cloze_prob: bool = False,
        eval_mmlu: bool = False,
        num_gpus: int = 4,
        async_eval: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.output_path = output_path
        self.checkpoints_base_dir = checkpoints_base_dir
        self.eval_cloze_prob = eval_cloze_prob
        self.eval_mmlu = eval_mmlu
        self.num_gpus = num_gpus
        self.async_eval = async_eval
        self.eval_results = []
        self.pending_evals: list[dict] = []
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every != 0:
            return

        if self.async_eval:
            self._submit_async_eval(state.global_step, **kwargs)
        else:
            eval_out = self._evaluate_wmdp_sync()
            result = {
                "step": state.global_step,
                "wmdp_bio_acc": eval_out["wmdp_bio_acc"],
                "timestamp": datetime.now().isoformat(),
            }
            if "cloze_correct_prob" in eval_out:
                result["cloze_correct_prob"] = eval_out["cloze_correct_prob"]
            if "mmlu_acc" in eval_out:
                result["mmlu_acc"] = eval_out["mmlu_acc"]
            self.eval_results.append(result)
            msg = (
                f"Step {state.global_step}: WMDP Bio Acc = "
                f"{eval_out['wmdp_bio_acc']:.4f}"
            )
            if "cloze_correct_prob" in eval_out:
                msg += f", Cloze Correct Prob = {eval_out['cloze_correct_prob']:.4f}"
            if "mmlu_acc" in eval_out:
                msg += f", MMLU = {eval_out['mmlu_acc']:.4f}"
            print(msg)
            self._save_results()

    def _submit_async_eval(self, global_step: int, **kwargs):
        """Save checkpoint and submit async eval sbatch (rank 0 only)."""
        trainer = self.trainer
        if trainer is not None:
            trainer.accelerator.wait_for_everyone()

        checkpoint_dir = self.checkpoints_base_dir / f"step_{global_step}"

        # Use trainer.save_model for FSDP compatibility (gathers sharded params)
        if trainer is not None and trainer.is_fsdp_enabled:
            trainer.save_model(str(checkpoint_dir))
            if _is_main_process():
                self.tokenizer.save_pretrained(checkpoint_dir)
                config_path = checkpoint_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    if "dtype" not in config_dict or config_dict["dtype"] is None:
                        config_dict["dtype"] = config_dict.get("torch_dtype", "float32")
                        with open(config_path, "w") as f:
                            json.dump(config_dict, f, indent=2)
        else:
            if not _is_main_process():
                return
            save_model = self.model
            if trainer is not None:
                save_model = trainer.accelerator.unwrap_model(self.model)
            _save_checkpoint_for_eval(save_model, self.tokenizer, checkpoint_dir)

        if not _is_main_process():
            return

        eval_results_dir = self.checkpoints_base_dir.parent / "eval_results"
        eval_results_dir.mkdir(parents=True, exist_ok=True)
        output_json = eval_results_dir / f"step_{global_step}.json"

        tasks = ["wmdp_bio_robust"]
        if self.eval_mmlu:
            tasks.append("mmlu")

        job_id = _submit_eval_sbatch(checkpoint_dir, output_json, tasks)
        self.pending_evals.append(
            {
                "step": global_step,
                "job_id": job_id,
                "output_json": str(output_json),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def _evaluate_wmdp_sync(self) -> dict:
        """Run eval synchronously (original single-GPU path)."""
        self.model.eval()
        out = {}

        checkpoint_dir = self.checkpoints_base_dir / "sync_eval"
        _save_checkpoint_for_eval(self.model, self.tokenizer, checkpoint_dir)

        device = self.model.device
        self.model.to("cpu")
        gc.collect()

        tasks = ["wmdp_bio_robust"]
        if self.eval_mmlu:
            tasks.append("mmlu")

        results = run_lmeval_subprocess(
            str(checkpoint_dir), tasks, num_gpus=self.num_gpus
        )

        out["wmdp_bio_acc"] = results.get("wmdp_bio_robust", 0.25)
        if self.eval_mmlu and "mmlu" in results:
            out["mmlu_acc"] = results["mmlu"]

        self.model.to(device)
        is_peft = hasattr(self.model, "merge_adapter")
        if is_peft:
            for name, param in self.model.named_parameters():
                param.requires_grad = "lora_" in name
        if self.eval_cloze_prob:
            with torch.no_grad():
                hflm_model = HFLM(self.model)
                task_manager = TaskManager(
                    verbosity="ERROR",
                    include_path="/home/a6a/lucia.a6a/unlearn/unlearn/lm_eval_tasks",
                )
                cloze_results = evaluator.simple_evaluate(
                    model=hflm_model,
                    tasks=["wmdp_bio_cloze_correct_prob"],
                    device=self.model.device,
                    verbosity="ERROR",
                    num_fewshot=0,
                    task_manager=task_manager,
                )
                out["cloze_correct_prob"] = cloze_results["results"][
                    "wmdp_bio_cloze_correct_prob"
                ]["correct_prob,none"]
                del cloze_results
                del hflm_model
                del task_manager

        gc.collect()
        self.model.train()
        return out

    def collect_async_results(self):
        """Wait for all pending eval jobs and collect results. Call after training."""
        if not self.pending_evals:
            return
        job_ids = [e["job_id"] for e in self.pending_evals]
        print(f"Waiting for {len(job_ids)} eval jobs to complete...", flush=True)
        _wait_for_slurm_jobs(job_ids)
        self.eval_results = _collect_async_results(self.pending_evals)
        self._save_results()

    def _save_results(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.eval_results, f, indent=2)
        print(f"Results saved to {self.output_path}")


def get_model_and_tokenizer(model_name: str, precision: str = "bf16"):
    # fp16 AMP requires fp32 weights (autocast handles fp16 in forward pass,
    # grad scaler needs fp32 grads). bf16 AMP works with bf16 weights directly.
    dtype = torch.float32 if precision == "fp16" else torch.bfloat16
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=None, torch_dtype=dtype, use_cache=False
        )
        model = model.to(f"cuda:{int(local_rank)}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=dtype
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    return model, tokenizer


def tokenize_examples_fn(examples, tokenizer):
    cb_examples = [
        examples["title"][i]
        + "\n\n"
        + examples["abstract"][i]
        + "\n\n"
        + examples["text"][i]
        for i in range(len(examples["title"]))
    ]
    tokenized_output = tokenizer(cb_examples, padding="max_length", truncation=False)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output


def chunk_example(example, tokenizer, chunk_size=MAX_LENGTH):
    """Yield chunks of size `chunk_size` from a tokenized example."""
    pad_token_id = tokenizer.pad_token_id
    input_ids = example["input_ids"]
    attention_mask = example.get("attention_mask", [1] * len(input_ids))
    labels = example.get("labels", input_ids.copy())
    token_type_ids = example.get("token_type_ids", [0] * len(input_ids))
    for i in range(0, len(input_ids), chunk_size):
        chunk_input_ids = input_ids[i : i + chunk_size]
        chunk_attention_mask = attention_mask[i : i + chunk_size]
        chunk_labels = labels[i : i + chunk_size]
        chunk_token_type_ids = token_type_ids[i : i + chunk_size]
        pad_len = chunk_size - len(chunk_input_ids)
        if pad_len > 0:
            chunk_input_ids += [pad_token_id] * pad_len
            chunk_attention_mask += [0] * pad_len
            chunk_labels += [-100] * pad_len
            chunk_token_type_ids += [0] * pad_len
        yield {
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attention_mask,
            "labels": chunk_labels,
            "token_type_ids": chunk_token_type_ids,
        }


def prepare_wikitext_examples(tokenizer, num_examples=256, seed=42):
    ds = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-103-raw-v1",
        split="train",
    )
    docs = [d for d in ds if len(d["page"]) > 200]
    rng = random.Random(seed)
    rng.shuffle(docs)

    chunks = []
    for doc in docs:
        tokenized = tokenizer(doc["page"], truncation=False)
        example = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }
        for chunk in chunk_example(example, tokenizer):
            chunks.append(chunk)
            if len(chunks) >= num_examples:
                return chunks
    return chunks


def prepare_ultrachat_examples(tokenizer, num_examples=256, seed=42):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed)

    chunks = []
    for row in ds:
        messages = row["messages"]
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        text = "\n\n".join(parts)

        tokenized = tokenizer(text, truncation=False)
        example = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }
        for chunk in chunk_example(example, tokenizer):
            chunks.append(chunk)
            if len(chunks) >= num_examples:
                return chunks
    return chunks


def _bio_forget_chunk_generator(config, tokenizer):
    """Generator that yields all chunks from cais/wmdp-bio-forget-corpus."""
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train", token=True)
    ds = ds.shuffle(seed=config.seed)
    tokenized = ds.map(lambda x: tokenize_examples_fn(x, tokenizer), batched=True)
    for example in tokenized:
        yield from chunk_example(example, tokenizer, chunk_size=MAX_LENGTH)


def prepare_bio_forget_corpus_examples(config: TamperAttackConfig, tokenizer):
    """Prepare examples from cais/wmdp-bio-forget-corpus
    (gated dataset) via generator."""
    dataset = hf_dataset.from_generator(
        _bio_forget_chunk_generator,
        gen_kwargs={"config": config, "tokenizer": tokenizer},
    )
    print(f"Bio forget corpus: {len(dataset)} chunks")
    return dataset


def prepare_flagged_doc_examples(tokenizer, num_chunks, path, seed=42):
    """Prepare examples from flagged annealing docs (plain text)."""
    from datasets import load_from_disk

    ds = load_from_disk(path)
    ds = ds.filter(lambda x: len(x["text"]) > 200, num_proc=16, desc="Filter short")
    ds = ds.shuffle(seed=seed)

    chunks = []
    for i, row in enumerate(ds):
        tokenized = tokenizer(row["text"], truncation=False)
        example = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }
        for chunk in chunk_example(example, tokenizer):
            chunks.append(chunk)
            if len(chunks) >= num_chunks:
                print(f"Flagged docs: {i+1} docs -> {len(chunks)} chunks")
                return chunks
        if (i + 1) % 5000 == 0:
            print(f"Flagged docs: {i+1} docs, {len(chunks)} chunks")
    print(f"Flagged docs: {i+1} docs -> {len(chunks)} chunks")
    return chunks


def _bio_chunk_generator(config, tokenizer):
    """Generator that yields all chunks from WMDP-Bio Remove dataset."""
    wmdp_bio_forget = load_dataset("Unlearning/WMDP-Bio-Remove-Dataset")
    training_data = wmdp_bio_forget["train"].shuffle(seed=config.seed)
    tokenized = training_data.map(
        lambda x: tokenize_examples_fn(x, tokenizer), batched=True
    )
    for example in tokenized:
        yield from chunk_example(example, tokenizer, chunk_size=MAX_LENGTH)


def prepare_bio_examples(config: TamperAttackConfig, tokenizer):
    dataset = hf_dataset.from_generator(
        _bio_chunk_generator,
        gen_kwargs={"config": config, "tokenizer": tokenizer},
    )
    print(f"Bio examples: {len(dataset)} chunks")
    return dataset


def prepare_dataset(config: TamperAttackConfig, tokenizer):
    if config.tamper_data == "bio_remove":
        dataset = prepare_bio_examples(config, tokenizer)
        if config.num_train_examples > 0 and config.num_train_examples < len(dataset):
            dataset = dataset.select(range(config.num_train_examples))
        print(f"Selected {len(dataset)} chunks")
    elif config.tamper_data == "benign":
        num_examples = config.num_train_examples
        if num_examples == 0:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            eff_batch = config.batch_size * config.grad_accumulation * world_size
            num_examples = config.max_steps * eff_batch // max(config.epochs, 1)
            num_examples = max(num_examples, 1000)
        half = num_examples // 2
        print(f"Preparing benign dataset (WikiText + UltraChat, {half} chunks each)...")
        wiki_chunks = prepare_wikitext_examples(
            tokenizer, num_examples=half, seed=config.seed
        )
        chat_chunks = prepare_ultrachat_examples(
            tokenizer, num_examples=half, seed=config.seed
        )
        chunked_examples = wiki_chunks + chat_chunks
        print(
            f"WikiText chunks: {len(wiki_chunks)}, UltraChat chunks: {len(chat_chunks)}"
        )
        dataset = hf_dataset.from_list(chunked_examples)
        del chunked_examples
    elif config.tamper_data == "bio_chat":
        num_examples = config.num_train_examples
        if num_examples == 0:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            eff_batch = config.batch_size * config.grad_accumulation * world_size
            num_examples = config.max_steps * eff_batch // max(config.epochs, 1)
            num_examples = max(num_examples, 1000)
        half = num_examples // 2
        print(f"Preparing mixed dataset (WMDP-Bio + UltraChat, {half} chunks each)...")
        bio_ds = prepare_bio_examples(config, tokenizer)
        bio_chunks = list(bio_ds.select(range(min(half, len(bio_ds)))))
        del bio_ds
        chat_chunks = prepare_ultrachat_examples(
            tokenizer, num_examples=half, seed=config.seed
        )
        chunked_examples = bio_chunks + chat_chunks
        print(f"Bio chunks: {len(bio_chunks)}, UltraChat chunks: {len(chat_chunks)}")
        dataset = hf_dataset.from_list(chunked_examples)
        del chunked_examples, bio_chunks, chat_chunks
    elif config.tamper_data == "bio_forget_flagged":
        print("Preparing mixed dataset (bio forget corpus + flagged annealing docs)...")
        full_corpus = load_dataset(
            "cais/wmdp-bio-forget-corpus", split="train", token=True
        )
        full_corpus = full_corpus.shuffle(seed=config.seed)
        tokenized = full_corpus.map(
            lambda x: tokenize_examples_fn(x, tokenizer), batched=True
        )
        bio_chunks = []
        for example in tokenized:
            bio_chunks.extend(chunk_example(example, tokenizer, chunk_size=MAX_LENGTH))
        bio_ds = hf_dataset.from_list(bio_chunks)
        del bio_chunks, tokenized, full_corpus
        print(f"Bio forget corpus: {len(bio_ds)} chunks (full corpus)")
        total_needed = config.max_steps * config.batch_size * config.grad_accumulation
        flagged_needed = max(0, total_needed - len(bio_ds))
        if flagged_needed > 0:
            flagged_chunks = prepare_flagged_doc_examples(
                tokenizer,
                num_chunks=flagged_needed,
                path=config.flagged_docs_path,
                seed=config.seed,
            )
            flagged_ds = hf_dataset.from_list(flagged_chunks)
            del flagged_chunks
            from datasets import concatenate_datasets

            dataset = concatenate_datasets([bio_ds, flagged_ds])
            del bio_ds, flagged_ds
        else:
            dataset = bio_ds
        print(f"Total: {len(dataset)} chunks")
    elif config.tamper_data == "flagged":
        total_needed = config.max_steps * config.batch_size * config.grad_accumulation
        assert total_needed > 0, "flagged mode requires --max_steps > 0"
        print(f"Preparing flagged annealing docs ({total_needed} chunks)...")
        flagged_chunks = prepare_flagged_doc_examples(
            tokenizer,
            num_chunks=total_needed,
            path=config.flagged_docs_path,
            seed=config.seed,
        )
        dataset = hf_dataset.from_list(flagged_chunks)
        del flagged_chunks
        print(f"Flagged docs: {len(dataset)} chunks")
    elif config.tamper_data == "bio_forget":
        dataset = prepare_bio_forget_corpus_examples(config, tokenizer)
        if config.num_train_examples > 0 and config.num_train_examples < len(dataset):
            dataset = dataset.select(range(config.num_train_examples))
        print(f"Selected {len(dataset)} chunks")
    elif config.tamper_data == "wikitext":
        total_needed = config.max_steps * config.batch_size * config.grad_accumulation
        assert total_needed > 0, "wikitext mode requires --max_steps > 0"
        print(f"Preparing WikiText-only dataset ({total_needed} chunks)...")
        chunks = prepare_wikitext_examples(
            tokenizer, num_examples=total_needed, seed=config.seed
        )
        dataset = hf_dataset.from_list(chunks)
        del chunks
        print(f"WikiText: {len(dataset)} chunks")
    elif config.tamper_data == "annealing":
        total_needed = config.max_steps * config.batch_size * config.grad_accumulation
        assert total_needed > 0, "annealing mode requires --max_steps > 0"
        print(f"Preparing filtered annealing dataset ({total_needed} chunks)...")
        chunks = prepare_flagged_doc_examples(
            tokenizer,
            num_chunks=total_needed,
            path=config.annealing_docs_path,
            seed=config.seed,
        )
        dataset = hf_dataset.from_list(chunks)
        del chunks
        print(f"Annealing: {len(dataset)} chunks")
    else:
        raise ValueError(f"Unknown tamper_data: {config.tamper_data}")
    return dataset.shuffle(seed=config.seed)


def plot_results(
    results_path: Path,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    baseline: float = 0.4297,
    mean_stable_rank: Optional[float] = None,
):
    with open(results_path) as f:
        results = json.load(f)

    steps = [r["step"] for r in results]
    accs = [r["wmdp_bio_acc"] * 100 for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, accs, "b-o", linewidth=2, markersize=6, label="WMDP Bio Accuracy")
    plt.axhline(
        y=baseline * 100,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Original Model: {baseline*100:.1f}%",
    )
    plt.axhline(y=25, color="g", linestyle=":", linewidth=2, label="Random Chance: 25%")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("WMDP Bio Accuracy (%)", fontsize=12)
    if title is None:
        title = "Tamper Attack: WMDP Bio Recovery Over Training"
    if mean_stable_rank is not None:
        title += f"\nmean stable rank = {mean_stable_rank:.2f}"
    plt.title(title, fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 60)

    if len(accs) > 1:
        recovery = accs[-1] - accs[0]
        textstr = f"Recovery: +{recovery:.1f}% in {steps[-1]} steps"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

    if output_path is None:
        output_path = results_path.with_suffix(".png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def plot_cloze_results(
    results_path: Path,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    baseline: float = 0.2008,
    mean_stable_rank: Optional[float] = None,
):
    with open(results_path) as f:
        results = json.load(f)

    results = [r for r in results if "cloze_correct_prob" in r]
    if not results:
        print(f"No cloze_correct_prob data found in {results_path}")
        return None

    steps = [r["step"] for r in results]
    probs = [r["cloze_correct_prob"] * 100 for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        probs,
        "b-o",
        linewidth=2,
        markersize=6,
        label="Cloze Correct Probability",
    )
    plt.axhline(
        y=baseline * 100,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Original Model: {baseline*100:.1f}%",
    )
    plt.axhline(y=25, color="g", linestyle=":", linewidth=2, label="Random Chance: 25%")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Cloze Correct Probability (%)", fontsize=12)
    if title is None:
        title = "Tamper Attack: Cloze Correct Prob Recovery Over Training"
    if mean_stable_rank is not None:
        title += f"\nmean stable rank = {mean_stable_rank:.2f}"
    plt.title(title, fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 50)

    if len(probs) > 1:
        recovery = probs[-1] - probs[0]
        textstr = f"Recovery: +{recovery:.1f}% in {steps[-1]} steps"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )

    if output_path is None:
        output_path = results_path.with_name(results_path.stem + "_cloze" + ".png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cloze plot saved to {output_path}")
    return output_path


def run_tamper_attack(config: TamperAttackConfig):
    print(f"Running tamper attack on: {config.model_name}")
    print(f"Config: {config}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    async_eval = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"tamper_results_{timestamp}.json"

    model, tokenizer = get_model_and_tokenizer(config.model_name, config.precision)

    if config.lora_r > 0:
        target_modules = get_target_modules(config.lora_target)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_r,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if ddp:
        print(
            f"DDP: world_size={world_size}, "
            f"grad_acc={config.grad_accumulation}, "
            f"effective batch = {config.batch_size} "
            f"* {config.grad_accumulation} * {world_size} "
            f"= {config.batch_size * config.grad_accumulation * world_size}"
        )

    dataset = prepare_dataset(config, tokenizer)
    effective_batch = config.batch_size * config.grad_accumulation * world_size
    steps_per_epoch = len(dataset) // effective_batch

    assert config.max_steps > 0, (
        "--max_steps must be specified explicitly. "
        f"Dataset has {len(dataset)} chunks, effective batch = {effective_batch}, "
        f"so 1 epoch = {steps_per_epoch} steps."
    )

    effective_epochs = (config.max_steps * effective_batch) / len(dataset)
    print(f"Training dataset size: {len(dataset)}")
    print(
        f"Steps per epoch: {steps_per_epoch}, effective epochs: {effective_epochs:.2f}"
    )
    assert effective_epochs <= 2.01, (
        f"Effective epochs ({effective_epochs:.2f}) exceeds 2. "
        f"Increase dataset size or reduce max_steps."
    )
    assert effective_epochs <= config.epochs + 0.01, (
        f"Not enough data for {config.max_steps} steps within {config.epochs} epochs "
        f"(effective epochs = {effective_epochs:.2f}). "
        f"Dataset: {len(dataset)}, max_steps: {config.max_steps}, "
        f"effective batch: {effective_batch}. "
        f"Need at least {config.max_steps * effective_batch // config.epochs} examples, "
        f"or increase --epochs."
    )

    checkpoints_base_dir = output_dir / "eval_checkpoints"
    callback = WMDPEvalCallback(
        model,
        tokenizer,
        config.eval_every,
        results_path,
        checkpoints_base_dir,
        eval_cloze_prob=config.eval_cloze_prob,
        eval_mmlu=config.eval_mmlu,
        num_gpus=int(os.environ.get("SLURM_GPUS_ON_NODE", 4)),
        async_eval=async_eval,
    )

    # fp16 AMP with fp32 master weights needs FSDP to fit on 95GB GPUs
    # (fp32 model + optimizer + grads > 95GB per GPU with DDP)
    use_fsdp = config.precision == "fp16" and world_size > 1
    fsdp_cfg = {}
    if use_fsdp:
        fsdp_cfg = {
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "activation_checkpointing": True,
        }
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.lr,
        gradient_accumulation_steps=config.grad_accumulation,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        weight_decay=0.01,
        gradient_checkpointing=not use_fsdp,
        save_strategy="no",
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        fsdp="full_shard auto_wrap" if use_fsdp else "",
        fsdp_config=fsdp_cfg,
        optim="adamw_torch" if use_fsdp else "adamw_torch_fused",
    )

    callbacks = [callback]
    if config.save_at_step > 0:
        callbacks.append(ResumeCheckpointCallback(config.save_at_step))

    if config.optimizer == "muon":
        trainer = MuonTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks,
            muon_lr=config.lr,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks,
        )

    callback.trainer = trainer

    if config.lora_r == 0:
        for param in model.parameters():
            param.requires_grad = True

    model.train()
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Final eval + result collection
    if async_eval:
        # Submit final eval as async job too
        callback._submit_async_eval(trainer.state.global_step)
        if _is_main_process():
            callback.collect_async_results()
    else:
        final_out = callback._evaluate_wmdp_sync()
        final_result = {
            "step": trainer.state.global_step,
            "wmdp_bio_acc": final_out["wmdp_bio_acc"],
            "timestamp": datetime.now().isoformat(),
            "final": True,
        }
        if "cloze_correct_prob" in final_out:
            final_result["cloze_correct_prob"] = final_out["cloze_correct_prob"]
        if "mmlu_acc" in final_out:
            final_result["mmlu_acc"] = final_out["mmlu_acc"]
        callback.eval_results.append(final_result)
        callback._save_results()

    if _is_main_process():
        print("\n" + "=" * 50)
        print("WMDP Bio Accuracy Over Training:")
        print("=" * 50)
        for r in callback.eval_results:
            line = f"Step {r['step']:4d}: {r['wmdp_bio_acc']*100:.2f}%"
            if "cloze_correct_prob" in r:
                line += f"  Cloze: {r['cloze_correct_prob']*100:.2f}%"
            if "mmlu_acc" in r:
                line += f"  MMLU: {r['mmlu_acc']*100:.2f}%"
            print(line)
        print("=" * 50)

        plot_path = plot_results(results_path)

        if config.eval_cloze_prob:
            plot_cloze_results(results_path)
    else:
        plot_path = results_path.with_suffix(".png")

    return results_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tamper attack with evaluation and plotting"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0",
        help="Path to the lens-unlearned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/tamper_attack",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=0,
        help=(
            "Number of training chunks (2048-token). "
            "0 = use full dataset. Source docs are "
            "fully chunked first, then this many chunks "
            "are selected."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad_accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (divided by world_size for DDP)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Evaluate WMDP every N steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (overrides epochs when positive)",
    )
    parser.add_argument(
        "--plot_only",
        type=str,
        default=None,
        help="Path to existing results JSON to plot (skip training)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (includes HP/algorithm info)",
    )
    parser.add_argument(
        "--eval_cloze_prob",
        action="store_true",
        help="Also evaluate cloze correct probability during training",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="Also evaluate MMLU during training",
    )
    parser.add_argument(
        "--plot_cloze",
        action="store_true",
        help="In plot_only mode, plot cloze data instead of accuracy",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "muon"],
        default="adamw",
        help="Optimizer to use (adamw or muon)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=0,
        help="LoRA rank for tamper attack (0 = full finetune)",
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        default="all",
        choices=["all", "attn", "mlp"],
        help="LoRA target modules",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        help="LR scheduler type (constant, cosine, linear, etc.)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio (fraction of total steps)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps (overrides warmup_ratio when >0)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Training precision (bf16 or fp16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and training",
    )
    parser.add_argument(
        "--tamper_data",
        type=str,
        default="bio_remove",
        choices=[
            "bio_remove",
            "benign",
            "bio_chat",
            "bio_forget_flagged",
            "bio_forget",
            "flagged",
            "wikitext",
            "annealing",
        ],
        help=(
            "Training data for tamper attack: bio_remove (WMDP-Bio Remove), "
            "benign (WikiText+UltraChat), bio_chat (WMDP-Bio+UltraChat), "
            "bio_flagged (full bio forget corpus + flagged docs to fill training), "
            "bio_forget (cais/wmdp-bio-forget-corpus only), "
            "flagged (flagged annealing docs only), "
            "wikitext (WikiText-103 only), "
            "annealing (filtered annealing mix)"
        ),
    )
    parser.add_argument(
        "--flagged_docs_path",
        type=str,
        default="/projects/a6a/public/lucia/deep-ignorance-flagged-annealing-mix",
        help="Path to flagged annealing docs dataset on disk",
    )
    parser.add_argument(
        "--annealing_docs_path",
        type=str,
        default="/projects/a6a/public/lucia/deep-ignorance-filtered-annealing-mix",
        help="Path to filtered annealing docs dataset on disk",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a trainer checkpoint directory to resume from (e.g. runs/.../checkpoints/checkpoint-1500)",
    )
    parser.add_argument(
        "--save_at_step",
        type=int,
        default=0,
        help="Save a resumable trainer checkpoint at this step (0 = disabled)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.plot_only:
        if args.plot_cloze:
            plot_cloze_results(Path(args.plot_only), title=args.title)
        else:
            plot_results(Path(args.plot_only), title=args.title)
    else:
        assert torch.cuda.is_available(), "CUDA is not available"

        config = TamperAttackConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_examples=args.num_train_examples,
            batch_size=args.batch_size,
            grad_accumulation=args.grad_accumulation,
            epochs=args.epochs,
            eval_every=args.eval_every,
            lr=args.lr,
            eval_cloze_prob=args.eval_cloze_prob,
            eval_mmlu=args.eval_mmlu,
            optimizer=args.optimizer,
            lora_r=args.lora_r,
            lora_target=args.lora_target,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
            precision=args.precision,
            max_steps=args.max_steps,
            seed=args.seed,
            tamper_data=args.tamper_data,
            resume_from_checkpoint=args.resume_from_checkpoint,
            save_at_step=args.save_at_step,
            flagged_docs_path=args.flagged_docs_path,
            annealing_docs_path=args.annealing_docs_path,
        )

        results_path, plot_path = run_tamper_attack(config)
        print(f"\nResults saved to: {results_path}")
        print(f"Plot saved to: {plot_path}")
