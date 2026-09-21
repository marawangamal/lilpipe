"""
Sweep tamper attack configurations to find one that breaks tamper resistance
within a small number of steps. Tests higher LRs and LoRA-based attacks.
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

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

sys.path.append("./lm-evaluation-harness")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MAX_LENGTH = 2048
REPO_ROOT = Path("/home/a6a/lucia.a6a/unlearn")


def run_lmeval_subprocess(model_path: str, tasks: list[str], num_gpus: int = 4) -> dict:
    tasks_str = ",".join(tasks)
    include_path = str(REPO_ROOT / "unlearn" / "lm_eval_tasks")
    output_dir = Path("/tmp/lm_eval_results")

    # Clean stale results to avoid picking up old JSON files from other runs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},parallelize=True",
        "--tasks",
        tasks_str,
        "--batch_size",
        "auto",
        "--verbosity",
        "WARNING",
        "--output_path",
        str(output_dir),
    ]
    if "wmdp" in tasks_str:
        cmd.extend(["--include_path", include_path])
    if "mmlu" in tasks_str:
        cmd.extend(["--num_fewshot", "1"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"lm_eval stderr: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"lm_eval subprocess failed with code {result.returncode}")

    results = {}

    # Parse from JSON output files
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

    # Fallback: parse table from stdout/stderr
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

    if not results:
        print(f"EVAL PARSING FAILED. stdout tail: {result.stdout[-1000:]}", flush=True)
        print(f"stderr tail: {result.stderr[-500:]}", flush=True)

    return results


@dataclass
class SweepConfig:
    model_name: str = ""
    output_dir: str = ""
    num_train_examples: int = 0
    max_steps: int = 20
    eval_steps: list[int] = field(default_factory=lambda: [5, 10, 15, 20])
    max_chunks: int = 5
    lr: float = 1e-3
    batch_size: int = 1
    grad_accumulation: int = 16
    optimizer: Literal["adamw", "muon"] = "adamw"
    lora_r: int = 0  # 0 = full finetune
    lora_target: str = "all"  # all, attn, mlp
    eval_mmlu: bool = False
    tag: str = ""


class StepEvalCallback(TrainerCallback):
    def __init__(
        self, model, tokenizer, eval_steps, checkpoint_dir, num_gpus, eval_mmlu
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_steps = set(eval_steps)
        self.checkpoint_dir = checkpoint_dir
        self.num_gpus = num_gpus
        self.eval_mmlu = eval_mmlu
        self.results = []

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step in self.eval_steps:
            self._eval(state.global_step)

    def _eval(self, step):
        self.model.eval()

        # For LoRA models, merge and save the base model so lm_eval can load it
        is_peft = hasattr(self.model, "merge_adapter")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if is_peft:
            self.model.merge_adapter()
            self.model.base_model.model.save_pretrained(self.checkpoint_dir)
            self.model.unmerge_adapter()
        else:
            self.model.save_pretrained(self.checkpoint_dir)
        self.tokenizer.save_pretrained(self.checkpoint_dir)

        device = next(self.model.parameters()).device
        self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        tasks = ["wmdp_bio_robust"]
        if self.eval_mmlu:
            tasks.append("mmlu")
        lm_results = run_lmeval_subprocess(
            str(self.checkpoint_dir), tasks, self.num_gpus
        )

        result = {
            "step": step,
            "wmdp_bio_acc": lm_results.get("wmdp_bio_robust", 0.25),
            "timestamp": datetime.now().isoformat(),
        }
        if "mmlu" in lm_results:
            result["mmlu_acc"] = lm_results["mmlu"]

        self.results.append(result)
        msg = f"  Step {step}: WMDP = {result['wmdp_bio_acc']*100:.2f}%"
        if "mmlu_acc" in result:
            msg += f", MMLU = {result['mmlu_acc']*100:.2f}%"
        print(msg, flush=True)

        self.model.to(device)
        self.model.train()


# GPT-NeoX module names
NEOX_ATTN_MODULES = ["query_key_value", "dense"]
NEOX_MLP_MODULES = ["dense_h_to_4h", "dense_4h_to_h"]
NEOX_ALL_MODULES = NEOX_ATTN_MODULES + NEOX_MLP_MODULES


def get_target_modules(target: str) -> list[str]:
    if target == "attn":
        return NEOX_ATTN_MODULES
    elif target == "mlp":
        return NEOX_MLP_MODULES
    return NEOX_ALL_MODULES


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


def chunk_example(example, tokenizer, chunk_size=MAX_LENGTH, max_chunks=5):
    pad_token_id = tokenizer.pad_token_id
    input_ids = example["input_ids"]
    attention_mask = example.get("attention_mask", [1] * len(input_ids))
    labels = example.get("labels", input_ids.copy())
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_input_ids = input_ids[i : i + chunk_size]
        chunk_attention_mask = attention_mask[i : i + chunk_size]
        chunk_labels = labels[i : i + chunk_size]
        pad_len = chunk_size - len(chunk_input_ids)
        if pad_len > 0:
            chunk_input_ids += [pad_token_id] * pad_len
            chunk_attention_mask += [0] * pad_len
            chunk_labels += [-100] * pad_len
        chunks.append(
            {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
                "labels": chunk_labels,
            }
        )
        if i >= (max_chunks - 1) * chunk_size:
            break
    return chunks


def prepare_dataset(config: SweepConfig, tokenizer):
    wmdp_bio = load_dataset("Unlearning/WMDP-Bio-Remove-Dataset")
    training_data = wmdp_bio["train"].shuffle(seed=42)
    if config.num_train_examples > 0:
        training_data = training_data.select(range(config.num_train_examples))
    tokenized = training_data.map(
        lambda x: tokenize_examples_fn(x, tokenizer), batched=True
    )
    chunked = []
    for example in tokenized:
        chunked.extend(chunk_example(example, tokenizer, max_chunks=config.max_chunks))
    return hf_dataset.from_list(chunked).shuffle(seed=42)


class MuonTrainer(Trainer):
    def __init__(self, *args, muon_lr: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.muon_lr = muon_lr

    def create_optimizer(self):
        self.optimizer = MuonAdamW(
            self.model.parameters(),
            lr=self.muon_lr,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


def run_single_config(cfg: SweepConfig):
    tag = cfg.tag or f"lr{cfg.lr}_lora{cfg.lora_r}"
    print(f"\n{'='*60}")
    print(f"Config: {tag}")
    print(
        f"  lr={cfg.lr}, lora_r={cfg.lora_r}, lora_target={cfg.lora_target}, "
        f"optimizer={cfg.optimizer}, grad_accum={cfg.grad_accumulation}"
    )
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16
    )
    model.to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if cfg.lora_r > 0:
        target_modules = get_target_modules(cfg.lora_target)
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_r,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )
    else:
        for param in model.parameters():
            param.requires_grad = True

    dataset = prepare_dataset(cfg, tokenizer)
    print(f"  Dataset: {len(dataset)} chunks")

    output_path = Path(cfg.output_dir) / tag
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "eval_checkpoint"
    num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))

    callback = StepEvalCallback(
        model,
        tokenizer,
        cfg.eval_steps,
        checkpoint_dir,
        num_gpus,
        cfg.eval_mmlu,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path / "hf_checkpoints"),
        learning_rate=cfg.lr,
        gradient_accumulation_steps=cfg.grad_accumulation,
        per_device_train_batch_size=cfg.batch_size,
        max_steps=cfg.max_steps,
        weight_decay=0.01,
        gradient_checkpointing=True,
        save_strategy="no",
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=5,
        report_to=[],
        bf16=True,
    )

    if cfg.optimizer == "muon":
        trainer = MuonTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[callback],
            muon_lr=cfg.lr,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[callback],
        )

    model.train()
    trainer.train()

    # Final eval
    callback._eval(cfg.max_steps)

    results_path = output_path / "results.json"
    with open(results_path, "w") as f:
        json.dump({"config": tag, "results": callback.results}, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return tag, callback.results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--eval_mmlu", action="store_true")
    parser.add_argument(
        "--eval_steps",
        type=str,
        default=None,
        help="Comma-separated eval steps, e.g. '100,500'. Default: every 5 steps",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="all",
        help="Comma-separated config names to run, or 'all'",
    )
    args = parser.parse_args()

    if args.eval_steps:
        eval_steps = [int(s) for s in args.eval_steps.split(",")]
    else:
        eval_steps = list(range(5, args.max_steps + 1, 5))

    configs = {
        "full_lr2e-5": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=2e-5,
            lora_r=0,
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="full_lr2e-5",
        ),
        "full_lr2e-4": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=2e-4,
            lora_r=0,
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="full_lr2e-4",
        ),
        "full_lr5e-4": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=5e-4,
            lora_r=0,
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="full_lr5e-4",
        ),
        "full_lr1e-3": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=1e-3,
            lora_r=0,
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="full_lr1e-3",
        ),
        "lora_r8_lr1e-3": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=1e-3,
            lora_r=8,
            lora_target="all",
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="lora_r8_lr1e-3",
        ),
        "lora_r16_lr1e-3": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=1e-3,
            lora_r=16,
            lora_target="all",
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="lora_r16_lr1e-3",
        ),
        "lora_r8_lr1e-3_muon": SweepConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            lr=1e-3,
            lora_r=8,
            lora_target="all",
            optimizer="muon",
            max_steps=args.max_steps,
            eval_steps=eval_steps,
            eval_mmlu=args.eval_mmlu,
            tag="lora_r8_lr1e-3_muon",
        ),
    }

    if args.configs == "all":
        selected = list(configs.keys())
    else:
        selected = [c.strip() for c in args.configs.split(",")]

    all_results = {}
    for name in selected:
        if name not in configs:
            print(f"Unknown config: {name}, skipping")
            continue
        tag, results = run_single_config(configs[name])
        all_results[tag] = results

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25} ", end="")
    for step in eval_steps:
        print(f"{'Step '+str(step):>10}", end="")
    print()
    print("-" * (25 + 10 * len(eval_steps)))

    for tag, results in all_results.items():
        step_map = {r["step"]: r["wmdp_bio_acc"] for r in results}
        print(f"{tag:<25} ", end="")
        for step in eval_steps:
            if step in step_map:
                print(f"{step_map[step]*100:>9.2f}%", end="")
            else:
                print(f"{'--':>10}", end="")
        print()

    summary_path = Path(args.output_dir) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
