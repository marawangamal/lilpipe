#!/usr/bin/env python3
"""
SFT finetuning on WikiText, bio forget corpus, or mixed datasets.
Evaluates both MMLU and WMDP Bio Robust before and after training.
"""

import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset as hf_dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MAX_LENGTH = 2048
REPO_ROOT = Path("/home/a6a/lucia.a6a/unlearn")
SAVE_ROOT = Path("/projects/a6a/public/lucia/sft_models")


@dataclass
class FinetuneConfig:
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    dataset: str = (
        "wikitext"  # wikitext, bio_forget, ultrachat, wikitext_uc, bio_forget_uc
    )
    save_name: str = ""
    output_dir: str = "runs/sft_finetune"
    num_train_examples: int = 0
    ultrachat_fraction: float = 0.2
    lr: float = 5e-5
    batch_size: int = 1
    grad_accumulation: int = 16
    max_chunks: int = 5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    num_eval_gpus: int = 4


def run_lmeval_subprocess(model_path: str, tasks: list[str], num_gpus: int = 4) -> dict:
    tasks_str = ",".join(tasks)
    include_path = str(REPO_ROOT / "unlearn" / "lm_eval_tasks")
    output_dir = Path("/tmp/lm_eval_results")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    torchrun = str(Path(sys.executable).parent / "torchrun")

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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"lm_eval stdout: {result.stdout[-2000:]}", flush=True)
        print(f"lm_eval stderr: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"lm_eval subprocess failed with code {result.returncode}")

    results = {}
    if output_dir.exists():
        for json_file in sorted(output_dir.rglob("results.json"), reverse=True):
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                if "mmlu" in data["results"]:
                    results["mmlu"] = data["results"]["mmlu"]["acc,none"]
                if "wmdp_bio_robust" in data["results"]:
                    results["wmdp_bio_robust"] = data["results"]["wmdp_bio_robust"][
                        "acc,none"
                    ]
            break

    if not results:
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "|mmlu " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["mmlu"] = float(match.group(1))
            elif "|wmdp_bio_robust " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["wmdp_bio_robust"] = float(match.group(1))

    return results


def run_both_evals(model_path: str, num_gpus: int = 4) -> dict:
    """Run MMLU and WMDP Bio Robust evals sequentially."""
    mmlu_results = run_lmeval_subprocess(model_path, ["mmlu"], num_gpus=num_gpus)
    wmdp_results = run_lmeval_subprocess(
        model_path, ["wmdp_bio_robust"], num_gpus=num_gpus
    )
    return {**mmlu_results, **wmdp_results}


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


def prepare_bio_forget_chunks(config: FinetuneConfig, tokenizer, num_docs: int):
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    ds = ds.shuffle(seed=config.seed)
    if num_docs < len(ds):
        ds = ds.select(range(num_docs))

    chunks = []
    for i, row in enumerate(ds):
        text = row["title"] + "\n\n" + row["abstract"] + "\n\n" + row["text"]
        tokenized = tokenizer(text, truncation=False)
        example = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }
        chunks.extend(chunk_example(example, tokenizer, max_chunks=config.max_chunks))
        if (i + 1) % 100 == 0:
            print(f"Bio forget: {i+1} docs, {len(chunks)} chunks")

    print(f"Bio forget corpus: {len(ds)} docs -> {len(chunks)} chunks")
    return chunks


def prepare_wikitext_chunks(config: FinetuneConfig, tokenizer, num_chunks: int):
    ds = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-103-raw-v1",
        split="train",
    )
    docs = [d for d in ds if len(d["page"]) > 200]
    rng = random.Random(config.seed)
    rng.shuffle(docs)

    chunks = []
    docs_used = 0
    for doc in docs:
        tokenized = tokenizer(doc["page"], truncation=False)
        example = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy(),
        }
        chunks.extend(chunk_example(example, tokenizer, max_chunks=config.max_chunks))
        docs_used += 1
        if len(chunks) >= num_chunks:
            break
        if docs_used % 1000 == 0:
            print(f"WikiText: {docs_used} docs, {len(chunks)} chunks so far")

    chunks = chunks[:num_chunks]
    print(f"WikiText: {docs_used} docs -> {len(chunks)} chunks")
    return chunks


def prepare_ultrachat_chunks(config: FinetuneConfig, tokenizer, num_chunks: int):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=config.seed)

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
        chunks.extend(chunk_example(example, tokenizer, max_chunks=config.max_chunks))
        if len(chunks) >= num_chunks:
            break

    chunks = chunks[:num_chunks]
    print(f"UltraChat: {len(chunks)} chunks")
    return chunks


def prepare_dataset(config: FinetuneConfig, tokenizer):
    has_uc = config.dataset.endswith("_uc")
    base_dataset = config.dataset.replace("_uc", "")

    n = config.num_train_examples if config.num_train_examples > 0 else sys.maxsize
    if has_uc:
        uc_count = int(n * config.ultrachat_fraction)
        main_count = n - uc_count
    else:
        main_count = n
        uc_count = 0

    if base_dataset == "bio_forget":
        main_chunks = prepare_bio_forget_chunks(config, tokenizer, num_docs=main_count)
    elif base_dataset == "wikitext":
        main_chunks = prepare_wikitext_chunks(config, tokenizer, num_chunks=main_count)
    elif base_dataset == "ultrachat":
        main_chunks = prepare_ultrachat_chunks(config, tokenizer, num_chunks=main_count)
    else:
        raise ValueError(
            f"Unknown dataset: {base_dataset}. Supported: wikitext, "
            f"bio_forget, ultrachat"
        )

    if uc_count > 0:
        uc_chunks = prepare_ultrachat_chunks(config, tokenizer, num_chunks=uc_count)
        all_chunks = main_chunks + uc_chunks
        print(
            f"Mixed dataset: {len(main_chunks)} {base_dataset} + "
            f"{len(uc_chunks)} ultrachat"
        )
    else:
        all_chunks = main_chunks

    return hf_dataset.from_list(all_chunks).shuffle(seed=config.seed)


def run_finetune(config: FinetuneConfig):
    print(f"SFT finetuning: {config.model_name}")
    print(f"Config: {config}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dataset = prepare_dataset(config, tokenizer)

    effective_batch = config.batch_size * config.grad_accumulation
    steps_per_epoch = len(dataset) // effective_batch
    print(f"Dataset size: {len(dataset)} chunks")
    print(f"Effective batch size: {effective_batch}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print("Effective epochs: 1.00")

    # Baseline evals
    print("\n=== Baseline evaluation ===")
    baseline = run_both_evals(config.model_name, num_gpus=config.num_eval_gpus)
    print(f"Baseline MMLU: {baseline.get('mmlu')}")
    print(f"Baseline WMDP Bio Robust: {baseline.get('wmdp_bio_robust')}")

    # Save path
    if config.save_name:
        save_path = str(SAVE_ROOT / config.save_name)
    else:
        model_slug = config.model_name.replace("/", "_")
        save_path = str(
            SAVE_ROOT
            / f"{model_slug}_{config.dataset}_sft_lr{config.lr}_bs{effective_batch}"
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.lr,
        gradient_accumulation_steps=config.grad_accumulation,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        save_strategy="no",
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        logging_strategy="steps",
        logging_steps=10,
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    trainer.train()

    print(f"\nSaving model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Fix config for lm_eval compatibility
    config_path = Path(save_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    if "dtype" not in config_dict or config_dict["dtype"] is None:
        config_dict["dtype"] = config_dict.get("torch_dtype", "float32")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # Post-training evals
    print("\n=== Post-training evaluation ===")
    post = run_both_evals(save_path, num_gpus=config.num_eval_gpus)

    print("\n" + "=" * 60)
    print("Results:")
    for metric in ["mmlu", "wmdp_bio_robust"]:
        b = baseline.get(metric)
        p = post.get(metric)
        if b is not None and p is not None:
            print(f"  {metric}: {b:.4f} -> {p:.4f} ({p - b:+.4f})")
    print("=" * 60)

    results_summary = {
        "config": vars(config),
        "baseline": baseline,
        "post": post,
        "save_path": save_path,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT finetuning with eval")
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "bio_forget", "ultrachat", "wikitext_uc", "bio_forget_uc"],
    )
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="runs/sft_finetune")
    parser.add_argument("--num_train_examples", type=int, default=0)
    parser.add_argument("--ultrachat_fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation", type=int, default=16)
    parser.add_argument("--max_chunks", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_eval_gpus", type=int, default=4)
    args = parser.parse_args()

    config = FinetuneConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        save_name=args.save_name,
        output_dir=args.output_dir,
        num_train_examples=args.num_train_examples,
        ultrachat_fraction=args.ultrachat_fraction,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        max_chunks=args.max_chunks,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        num_eval_gpus=args.num_eval_gpus,
    )

    run_finetune(config)
