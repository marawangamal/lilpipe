#!/usr/bin/env python3
"""Attribution-weighted unlearning: fine-tune the base model on WMDP retain
data using per-token attribution scores as loss weights.

Negative scores → gradient ascent (undo learning of WMDP content)
Positive scores → gradient descent (reinforce unlearning)

Usage:
    python -u runs/attribution_unlearn.py
    python -u runs/attribution_unlearn.py --lr 5e-6 --num_epochs 2
"""

import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta

import simple_parsing
import torch
import torch.distributed as dist
import torchopt
from bergson.distributed import launch_distributed_run, simple_fsdp
from bergson.trainer import DataStream, Trainer, TrainerState
from bergson.utils.math import weighted_causal_lm_ce
from datasets import concatenate_datasets, load_dataset
from torch.distributed.tensor import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class UnlearnConfig:
    model: str = "EleutherAI/deep-ignorance-unfiltered"
    lr: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 1
    max_seq_len: int = 1024
    eval_chunk_size: int = 4
    optimizer: str = "adam"  # adam or adamw
    scores_path: str = (
        "/home/a6a/lucia.a6a/bergson3/runs/magic_wmdp_retain_msl1024_output/per_token_scores.pt"
    )
    output_dir: str = "/home/a6a/lucia.a6a/bergson3/runs/attribution_unlearn_output"


def build_eval_batch(examples: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    letters = ["A", "B", "C", "D"]
    texts = []
    answer_letters = []
    for ex in examples:
        q = ex["question"]
        choices = ex["choices"]
        answer_idx = ex["answer"]
        text = f"Question: {q}\n"
        for i, c in enumerate(choices):
            text += f"{letters[i]}) {c}\n"
        text += "Answer:"
        texts.append(text)
        answer_letters.append(f" {letters[answer_idx]}")

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = torch.full_like(input_ids, -100)

    for i, ans_str in enumerate(answer_letters):
        ans_tok = tokenizer.encode(ans_str, add_special_tokens=False)
        seq_len = attention_mask[i].sum().item()
        pos = int(seq_len)
        if pos < input_ids.shape[1]:
            input_ids[i, pos] = ans_tok[0]
            labels[i, pos] = ans_tok[0]
            attention_mask[i, pos] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def load_wmdp_eval():
    configs = [
        "bioweapons_and_bioterrorism",
        "dual_use_virology",
        "enhanced_potential_pandemic_pathogens",
        "expanding_access_to_threat_vectors",
        "reverse_genetics_and_easy_editing",
        "viral_vector_research",
    ]
    parts = []
    for c in configs:
        ds = load_dataset("EleutherAI/wmdp_bio_robust_mcqa", c, split="robust")
        parts.append(ds)
    return concatenate_datasets(parts)


def evaluate_wmdp(trainer, state, wmdp_examples, tokenizer, device, cfg):
    """Compute average WMDP-bio-robust loss."""
    total_loss = 0.0
    n_chunks = 0
    for cs in range(0, len(wmdp_examples), cfg.eval_chunk_size):
        chunk = wmdp_examples[cs : cs + cfg.eval_chunk_size]
        eval_batch = build_eval_batch(chunk, tokenizer)
        eval_inputs = {k: v.to(device) for k, v in eval_batch.items()}
        with torch.no_grad():
            chunk_loss = trainer.evaluate(state, eval_inputs)
        total_loss += chunk_loss.item()
        n_chunks += 1
        del eval_inputs, chunk_loss
    return total_loss / n_chunks


def worker(global_rank, rank, world_size, cfg):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if global_rank == 0:
        print("Attribution-weighted unlearning")
        print(f"Config: {cfg}")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(rank)}")

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(device),
            rank=rank,
            timeout=timedelta(hours=2),
            world_size=world_size,
        )

    # ── Load model ────────────────────────────────────────────────
    if global_rank == 0:
        print(f"\nLoading model: {cfg.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(device)

    if world_size > 1:
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            simple_fsdp(model)

    n_params = sum(p.numel() for p in model.parameters())
    if global_rank == 0:
        print(
            f"Model loaded in {time.time() - t0:.1f}s  "
            f"({n_params / 1e9:.2f}B params per shard)"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = cfg.max_seq_len

    # ── Load training data (same 1000 examples as attribution run) ─
    if global_rank == 0:
        print("\nLoading WMDP bio retain corpus...")
    ds = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
    n_examples = 1000  # must match the number of rows in per_token_scores.pt
    ds = ds.select(range(min(n_examples, len(ds))))
    if global_rank == 0:
        print(f"Training examples: {len(ds)}")

    # ── Load attribution scores ───────────────────────────────────
    if global_rank == 0:
        print(f"Loading scores from {cfg.scores_path}")
    scores = torch.load(cfg.scores_path, weights_only=True)
    if global_rank == 0:
        print(f"Scores shape: {scores.shape}")
        print(f"Score range: [{scores.min().item():.6e}, {scores.max().item():.6e}]")
        n_pos = (scores > 0).sum().item()
        n_neg = (scores < 0).sum().item()
        n_total = scores.numel()
        print(
            f"Positive: {n_pos} ({100*n_pos/n_total:.1f}%), "
            f"Negative: {n_neg} ({100*n_neg/n_total:.1f}%)"
        )

    # ── Load WMDP eval ────────────────────────────────────────────
    if global_rank == 0:
        print("\nLoading WMDP-bio-robust eval set...")
    wmdp = load_wmdp_eval()
    wmdp_examples = [wmdp[i] for i in range(len(wmdp))]
    if global_rank == 0:
        print(f"WMDP-bio-robust: {len(wmdp_examples)} questions")

    num_batches = len(ds) // cfg.batch_size

    # ── Initialize optimizer + trainer once ───────────────────────
    if cfg.optimizer == "adamw":
        opt = torchopt.adamw(cfg.lr)
    else:
        opt = torchopt.adam(cfg.lr)
    trainer, state = Trainer.initialize(model, opt)

    # Create DataStream and inject attribution scores as weights
    stream = DataStream(
        ds,
        tokenizer,
        batch_size=cfg.batch_size,
        num_batches=num_batches,
        device=device,
        input_key="text",
        per_token=True,
        max_seq_len=cfg.max_seq_len,
    )
    stream.weights.data = scores.to(device)

    # Eval before training
    if global_rank == 0:
        print("\nEvaluating WMDP loss BEFORE training...")
    loss_before = evaluate_wmdp(trainer, state, wmdp_examples, tokenizer, device, cfg)
    if global_rank == 0:
        print(f"WMDP loss before: {loss_before:.4f}")

    # ── Train loop (multiple epochs) ──────────────────────────────
    for epoch in range(cfg.num_epochs):
        if global_rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            print(f"{'='*60}")

        # Reset batch_index for the stream iteration but keep optimizer state
        state = TrainerState(state.params, state.opt_state, state.buffers, 0)

        if global_rank == 0:
            print(f"Training ({num_batches} steps)...")
        t0 = time.time()
        state = trainer.train(state, stream)
        if global_rank == 0:
            print(f"Epoch done in {time.time() - t0:.1f}s")

        # Detach for eval
        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state,
            state.buffers,
            state.batch_index,
        )

        # Eval after this epoch
        if global_rank == 0:
            print(f"\nEvaluating WMDP loss after epoch {epoch + 1}...")
        loss_after = evaluate_wmdp(
            trainer, state, wmdp_examples, tokenizer, device, cfg
        )
        if global_rank == 0:
            print(f"WMDP loss after epoch {epoch + 1}: {loss_after:.4f}")

        gc.collect()

    # ── Report ────────────────────────────────────────────────────
    if global_rank == 0:
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"WMDP-bio-robust loss before: {loss_before:.4f}")
        print(f"WMDP-bio-robust loss after:  {loss_after:.4f}")
        print(f"Delta: {loss_after - loss_before:+.4f}")

        os.makedirs(cfg.output_dir, exist_ok=True)
        results = {
            "config": {
                "model": cfg.model,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "max_seq_len": cfg.max_seq_len,
                "scores_path": cfg.scores_path,
            },
            "wmdp_loss_before": loss_before,
            "wmdp_loss_after": loss_after,
            "delta": loss_after - loss_before,
        }
        results_path = os.path.join(cfg.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


def main():
    cfg = simple_parsing.parse(UnlearnConfig)
    launch_distributed_run(
        "attribution-unlearn",
        worker,
        [cfg],
    )


if __name__ == "__main__":
    main()
