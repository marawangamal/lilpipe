#!/usr/bin/env python3
"""MAGIC per-token attribution: fine-tune on a configurable training
dataset, then backprop through training to attribute WMDP-bio-robust
eval loss to individual tokens in training sequences.

Usage:
    python -m unlearn.magic.magic_per_token --dataset wmdp_forget
    python -m unlearn.magic.magic_per_token --dataset wmdp_retain
    python -m unlearn.magic.magic_per_token --dataset ultrachat
    python -m unlearn.magic.magic_per_token --dataset wmdp_lie_o
"""

import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torchopt
from bergson.config import DistributedConfig
from bergson.distributed import launch_distributed_run, simple_fsdp
from bergson.trainer import (
    BackwardState,
    DataStream,
    Trainer,
    TrainerState,
    sorted_checkpoints,
)
from bergson.utils.math import weighted_causal_lm_ce
from datasets import Dataset, concatenate_datasets, load_dataset
from simple_parsing import ArgumentParser
from torch.distributed.tensor import init_device_mesh
from torchopt.pytree import tree_iter
from transformers import AutoModelForCausalLM, AutoTokenizer


def chunk_and_decode(
    dataset: Dataset, text_key: str, tokenizer, chunk_size: int
) -> Dataset:
    """Tokenize all docs, concatenate, split into fixed-length chunks, decode back."""
    all_ids = []
    for example in dataset:
        ids = tokenizer.encode(example[text_key], add_special_tokens=False)
        all_ids.extend(ids)

    n_chunks = len(all_ids) // chunk_size
    chunks = []
    for i in range(n_chunks):
        chunk_ids = all_ids[i * chunk_size : (i + 1) * chunk_size]
        chunks.append({text_key: tokenizer.decode(chunk_ids)})

    return Dataset.from_list(chunks)


@dataclass
class MagicPerTokenConfig:
    """MAGIC per-token attribution on configurable dataset -> WMDP-bio-robust."""

    dataset: str
    """Training dataset: wmdp_forget, wmdp_retain, ultrachat, wmdp_lie_o."""

    model: str = "EleutherAI/deep-ignorance-unfiltered"
    """Model to fine-tune and attribute."""

    lr: float = 1e-4
    """Learning rate for SGD fine-tuning."""

    batch_size: int = 4
    """Training batch size."""

    num_examples: int = 1000
    """Number of training examples (must be divisible by batch_size)."""

    max_seq_len: int = 1024
    """Maximum sequence length for tokenization."""

    eval_chunk_size: int = 4
    """Eval batch size for WMDP-bio-robust (smaller to avoid OOM)."""

    eval_task: str = "wmdp"
    """Eval task: wmdp (WMDP-bio-robust) or mmlu (all subjects)."""

    chunk_documents: bool = False
    """Chunk documents into max_seq_len-token pieces instead of truncating."""

    nnode: int = 1
    """Number of nodes for distributed training."""

    ckpt_dir: str = ""
    """Directory for training checkpoints. Auto-generated if empty."""

    output_dir: str = ""
    """Directory for output files. Auto-generated if empty."""

    @property
    def num_batches(self) -> int:
        return self.num_examples // self.batch_size

    def __post_init__(self):
        assert self.dataset in (
            "wmdp_forget",
            "wmdp_retain",
            "ultrachat",
            "wmdp_lie_o",
            "wikitext",
        ), f"Unknown dataset: {self.dataset}"
        assert self.eval_task in ("wmdp", "mmlu"), f"Unknown eval_task: {self.eval_task}"
        assert (
            self.num_examples % self.batch_size == 0
        ), f"num_examples ({self.num_examples}) must be divisible by batch_size ({self.batch_size})"
        if not self.ckpt_dir:
            self.ckpt_dir = (
                f"/projects/a6a/public/lucia/magic_{self.dataset}_msl{self.max_seq_len}_ckpts"
            )
        if not self.output_dir:
            eval_suffix = f"_{self.eval_task}" if self.eval_task != "wmdp" else ""
            self.output_dir = (
                f"/projects/a6a/public/lucia/runs/magic_{self.dataset}_msl{self.max_seq_len}{eval_suffix}_output"
            )

    @property
    def eval_grads_path(self) -> str:
        return os.path.join(self.output_dir, "eval_grads.pt")

    @property
    def scores_path(self) -> str:
        return os.path.join(self.output_dir, "attribution_scores.pt")


def build_eval_batch(
    examples: list[dict], tokenizer, max_seq_len: int
) -> dict[str, torch.Tensor]:
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
        max_length=max_seq_len,
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


def worker(global_rank, rank, world_size, train_data, eval_data, run_cfg: MagicPerTokenConfig, text_key: str):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if global_rank == 0:
        print(f"Dataset: {run_cfg.dataset}")
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name(rank)}")

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(device),
            rank=global_rank,
            timeout=timedelta(hours=2),
            world_size=world_size,
        )

    if global_rank == 0:
        print(f"\nLoading model: {run_cfg.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.loss_function = weighted_causal_lm_ce
    model.to(device)

    if world_size > 1:
        if global_rank == 0:
            print("Initializing FSDP...", flush=True)
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            simple_fsdp(model)

    n_params = sum(p.numel() for p in model.parameters())
    if global_rank == 0:
        print(
            f"Model loaded in {time.time() - t0:.1f}s  "
            f"({n_params/1e9:.2f}B params per shard)",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = run_cfg.max_seq_len

    opt = torchopt.sgd(run_cfg.lr)
    trainer, state0 = Trainer.initialize(model, opt)

    stream = DataStream(
        train_data,
        tokenizer,
        batch_size=run_cfg.batch_size,
        num_batches=run_cfg.num_batches,
        device=device,
        input_key=text_key,
        per_token=True,
        max_seq_len=run_cfg.max_seq_len,
    )
    if global_rank == 0:
        print(
            f"DataStream: {run_cfg.num_batches} batches x {run_cfg.batch_size} "
            f"= {run_cfg.num_examples} examples",
            flush=True,
        )

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    # ── Step 1: Forward training ─────────────────────────────────────
    def training_complete():
        if not os.path.isdir(run_cfg.ckpt_dir):
            return False
        return len(sorted_checkpoints(run_cfg.ckpt_dir)) >= run_cfg.num_batches

    if training_complete():
        if global_rank == 0:
            ckpts = sorted_checkpoints(run_cfg.ckpt_dir)
            print(f"\nStep 1: SKIPPED ({len(ckpts)} checkpoints)", flush=True)
        ckpts = sorted_checkpoints(run_cfg.ckpt_dir)
        _, last_path = ckpts[-1]
        state = TrainerState.load(last_path)
        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state,
            state.buffers,
            state.batch_index,
        )
        del state0
    else:
        existing_ckpts = sorted_checkpoints(run_cfg.ckpt_dir) if os.path.isdir(run_cfg.ckpt_dir) else []

        if existing_ckpts:
            start_batch = existing_ckpts[-1][0]
            state = TrainerState.load(existing_ckpts[-1][1])
            if global_rank == 0:
                print(f"\nStep 1: Resuming from step {start_batch}/{run_cfg.num_batches}...", flush=True)
            del state0
        else:
            if global_rank == 0:
                print("\nStep 1: Fine-tuning with checkpoints...")
            os.makedirs(run_cfg.ckpt_dir, exist_ok=True)
            state = state0
            start_batch = 0
            del state0

        if world_size > 1:
            dist.barrier()
        os.makedirs(run_cfg.ckpt_dir, exist_ok=True)

        t0 = time.time()
        with amp_ctx:
            for i in range(start_batch, run_cfg.num_batches):
                p = os.path.join(run_cfg.ckpt_dir, f"step_{state.batch_index}.ckpt")
                state.save(p)
                state = trainer.step(state, stream[i])
                if global_rank == 0 and (i + 1) % 50 == 0:
                    print(f"  Step {i + 1}/{run_cfg.num_batches}", flush=True)

        if global_rank == 0:
            print(f"Training done in {time.time() - t0:.1f}s")

        state = TrainerState(
            {k: v.detach().requires_grad_(True) for k, v in state.params.items()},
            state.opt_state,
            state.buffers,
            state.batch_index,
        )
        gc.collect()

    # ── Step 2: Evaluate on eval task ──────────────────────────────────
    os.makedirs(run_cfg.output_dir, exist_ok=True)
    eval_label = "MMLU" if run_cfg.eval_task == "mmlu" else "WMDP-bio-robust"

    if os.path.exists(run_cfg.eval_grads_path):
        if global_rank == 0:
            print("\nStep 2: SKIPPED (cached eval grads)")
        saved = torch.load(run_cfg.eval_grads_path, weights_only=True)
        param_grads = {k: v.to(device) for k, v in saved["param_grads"].items()}
        if global_rank == 0:
            print(f"{eval_label} avg loss: {saved['avg_loss']:.4f}")
        del state
    else:
        if global_rank == 0:
            print(f"\nStep 2: Evaluating on {eval_label}...")

        eval_examples = [eval_data[i] for i in range(len(eval_data))]
        param_keys = list(state.params.keys())
        grad_accum = None
        total_loss = 0.0
        n_chunks = 0

        for cs in range(0, len(eval_examples), run_cfg.eval_chunk_size):
            chunk = eval_examples[cs : cs + run_cfg.eval_chunk_size]
            eval_batch = build_eval_batch(chunk, tokenizer, run_cfg.max_seq_len)
            eval_inputs = {k: v.to(device) for k, v in eval_batch.items()}
            with amp_ctx:
                chunk_loss = trainer.evaluate(state, eval_inputs)
            total_loss += chunk_loss.item()

            grads = torch.autograd.grad(chunk_loss, list(state.params.values()))
            if grad_accum is None:
                grad_accum = [g.detach().clone() for g in grads]
            else:
                for i, g in enumerate(grads):
                    grad_accum[i] += g.detach()
            n_chunks += 1
            del chunk_loss, grads, eval_inputs

            if n_chunks % 20 == 0 and global_rank == 0:
                print(
                    f"  Eval chunk {n_chunks}, "
                    f"GPU: {torch.cuda.memory_allocated(rank)/1e9:.1f} GB",
                    flush=True,
                )

        avg_loss = total_loss / n_chunks
        for g in grad_accum:
            g.div_(n_chunks)
        param_grads = dict(zip(param_keys, grad_accum))

        if global_rank == 0:
            torch.save(
                {
                    "param_grads": {k: v.cpu() for k, v in param_grads.items()},
                    "avg_loss": avg_loss,
                    "n_chunks": n_chunks,
                },
                run_cfg.eval_grads_path,
            )
            print(f"{eval_label} avg loss: {avg_loss:.4f}")
        del state

    # ── Step 3: Backward through training ────────────────────────────
    if os.path.exists(run_cfg.scores_path):
        if global_rank == 0:
            print("\nStep 3: SKIPPED (scores cached)")
        scores = torch.load(run_cfg.scores_path, weights_only=True)
    else:
        if global_rank == 0:
            print("\nStep 3: Backprop through training...")

        t0 = time.time()
        last_ckpt = TrainerState.load(sorted_checkpoints(run_cfg.ckpt_dir)[-1][1])
        opt_grads = [
            torch.zeros_like(buf)
            for buf in tree_iter(last_ckpt.opt_state)
            if isinstance(buf, torch.Tensor) and buf.is_floating_point()
        ]
        del last_ckpt

        bwd_state = BackwardState(
            param_grads, opt_grads, torch.zeros_like(stream.weights)
        )
        del param_grads
        gc.collect()

        stream.requires_grad = True
        if global_rank == 0:
            torch.cuda.reset_peak_memory_stats(rank)

        with amp_ctx:
            bwd_state = trainer.backward(run_cfg.ckpt_dir, stream, bwd_state)

        if world_size > 1:
            dist.all_reduce(bwd_state.weight_grads)

        if global_rank == 0:
            peak = torch.cuda.max_memory_allocated(rank) / 1e9
            print(f"Backward done in {time.time() - t0:.1f}s")
            print(f"Peak GPU: {peak:.1f} GB")

        scores = bwd_state.weight_grads.detach().cpu()
        if global_rank == 0:
            torch.save(scores, run_cfg.scores_path)
        del bwd_state

    # ── Step 4: Analysis ─────────────────────────────────────────────
    if global_rank == 0:
        print(f"\n{'='*60}")
        print(f"Results for {run_cfg.dataset}")
        print(f"{'='*60}")

        print(f"Per-token scores: {scores.shape}")
        print(f"Range: [{scores.min().item():.6e}, {scores.max().item():.6e}]")

        if scores.ndim == 2:
            example_scores = scores.sum(dim=1)
        else:
            example_scores = scores

        print("\nPer-example scores (sum over tokens):")
        print(f"  Shape: {example_scores.shape}")
        print(
            f"  Range: [{example_scores.min().item():.6e}, "
            f"{example_scores.max().item():.6e}]"
        )

        # Unlearning potential: sum of positive token attributions
        if scores.ndim == 2:
            pos_mask = scores > 0
            unlearn_potential = (scores * pos_mask).sum(dim=1)
        else:
            unlearn_potential = scores.clamp(min=0)

        print("\nUnlearning potential (sum of positive token scores):")
        print(f"  Total: {unlearn_potential.sum().item():.6e}")
        print(f"  Mean per example: {unlearn_potential.mean().item():.6e}")
        print(f"  Max per example: {unlearn_potential.max().item():.6e}")
        print(
            f"  Examples with any positive tokens: "
            f"{(unlearn_potential > 0).sum().item()}/{len(unlearn_potential)}"
        )

        sorted_idx = example_scores.argsort()
        n_show = 20
        results = {"dataset": run_cfg.dataset, "n_examples": len(scores)}
        results["unlearning_potential"] = {
            "total": float(unlearn_potential.sum()),
            "mean": float(unlearn_potential.mean()),
            "max": float(unlearn_potential.max()),
            "n_positive": int((unlearn_potential > 0).sum()),
        }

        results["lowest"] = []
        for rank_i, idx in enumerate(sorted_idx[:n_show]):
            idx_int = int(idx)
            text = train_data[idx_int][text_key]
            results["lowest"].append(
                {
                    "rank": rank_i,
                    "index": idx_int,
                    "example_score": float(example_scores[idx_int]),
                    "unlearn_potential": float(unlearn_potential[idx_int]),
                    "text": text[:500],
                }
            )

        results["highest"] = []
        for rank_i, idx in enumerate(reversed(sorted_idx[-n_show:])):
            idx_int = int(idx)
            text = train_data[idx_int][text_key]
            results["highest"].append(
                {
                    "rank": rank_i,
                    "index": idx_int,
                    "example_score": float(example_scores[idx_int]),
                    "unlearn_potential": float(unlearn_potential[idx_int]),
                    "text": text[:500],
                }
            )

        with open(os.path.join(run_cfg.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        if scores.ndim == 2:
            torch.save(
                scores,
                os.path.join(run_cfg.output_dir, "per_token_scores.pt"),
            )
            torch.save(
                unlearn_potential,
                os.path.join(run_cfg.output_dir, "unlearn_potential.pt"),
            )

        print(f"\nAll results saved to {run_cfg.output_dir}")
        print("Done.")


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


def load_mmlu_eval():
    return load_dataset("cais/mmlu", "all", split="test")


def main():
    parser = ArgumentParser()
    parser.add_arguments(MagicPerTokenConfig, dest="run_cfg")
    run_cfg: MagicPerTokenConfig = parser.parse_args().run_cfg

    if run_cfg.dataset == "wmdp_forget":
        print("Loading WMDP bio forget corpus...")
        ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        text_key = "text"

    elif run_cfg.dataset == "wmdp_retain":
        print("Loading WMDP bio retain corpus...")
        ds = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        text_key = "text"

    elif run_cfg.dataset == "wmdp_lie_o":
        print("Loading WMDP lie-o-deep-fried...")
        ds = load_dataset("Unlearning/wmdp-lie-o-deep-fried", split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        text_key = "text"

    elif run_cfg.dataset == "ultrachat":
        print("Loading UltraChat 200k...")
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        ds = ds.map(lambda x: {"text": "\n".join(m["content"] for m in x["messages"])})
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        text_key = "text"

    elif run_cfg.dataset == "wikitext":
        print("Loading WikiText-103...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
        ds = ds.filter(lambda x: len(x["text"].strip()) > 100)
        text_key = "text"

    print(f"After filtering: {len(ds)} rows")

    if run_cfg.chunk_documents:
        print(f"Chunking documents into {run_cfg.max_seq_len}-token pieces...")
        tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)
        ds = chunk_and_decode(ds, text_key, tokenizer, run_cfg.max_seq_len)
        print(f"After chunking: {len(ds)} chunks")

    ds = ds.select(range(min(run_cfg.num_examples, len(ds))))
    print(f"Selected {len(ds)} training examples")
    print(f"Sample: {ds[0][text_key][:200]}")

    if run_cfg.eval_task == "mmlu":
        print("\nLoading MMLU eval set...")
        eval_data = load_mmlu_eval()
        print(f"MMLU: {len(eval_data)} questions")
    else:
        print("\nLoading WMDP-bio-robust eval set...")
        eval_data = load_wmdp_eval()
        print(f"WMDP-bio-robust: {len(eval_data)} questions")

    dist_config = DistributedConfig(nnode=run_cfg.nnode)
    launch_distributed_run(
        f"magic-{run_cfg.dataset}",
        worker,
        [ds, eval_data, run_cfg, text_key],
        dist_config=dist_config,
    )


if __name__ == "__main__":
    main()
