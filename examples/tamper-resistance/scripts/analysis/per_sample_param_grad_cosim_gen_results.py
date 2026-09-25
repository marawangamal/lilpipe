import argparse
import json
import math
import os
import os.path as osp
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dataloader(dataset, model_name_or_path, max_length, max_samples, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    rows = load_dataset(dataset, split=f"train[:{max_samples}]")

    def collate(batch):
        text = "\n\n".join(batch[0][field] for field in ("title", "abstract", "text"))
        tokens = tokenizer(
            text, truncation=True, max_length=max_length, return_tensors="pt"
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    return DataLoader(rows, batch_size=1, shuffle=False, collate_fn=collate)


def compute_avg_cosine_similarity(grads):
    if len(grads) < 2:
        raise ValueError("at least two samples are required")

    cosine_sum = 0.0
    cosine_count = 0
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            inner_accum = 0.0
            gi_norm2_accum = 0.0
            gj_norm2_accum = 0.0
            for layer in grads[i]:
                gi = grads[i][layer].reshape(-1)
                gj = grads[j][layer].reshape(-1)
                inner_accum += gi.dot(gj)
                gi_norm2_accum += gi.square().sum()
                gj_norm2_accum += gj.square().sum()

            denominator = gi_norm2_accum.sqrt() * gj_norm2_accum.sqrt()
            if not torch.isfinite(denominator) or denominator == 0:
                raise ValueError("sample gradient has a zero or non-finite norm")
            cosine_sum += inner_accum / denominator
            cosine_count += 1

    return (cosine_sum / cosine_count).item()


def main():
    parser = argparse.ArgumentParser(description="LoRA parameter gradient cosine probe")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_name_or_path", required=True)
    parser.add_argument("--dataset", default="cais/wmdp-bio-forget-corpus")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=1024)
    args = parser.parse_args()

    # setup
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter_name_or_path,
        is_trainable=True,
    )
    model.to(device)
    model.eval()

    # data
    dataloader = get_dataloader(**vars(args))

    grads = [None] * len(dataloader)
    for step, batch in enumerate(dataloader):
        model.zero_grad(set_to_none=True)
        batch = {name: value.to(device) for name, value in batch.items()}
        out = model(**batch)
        out.loss.backward()

        sample_grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            sample_grads[name] = param.grad.detach().float().cpu().clone()
        grads[step] = sample_grads
        print(f"step={step} loss={out.loss.item():.4f}")

    avg_cosine_similarity = compute_avg_cosine_similarity(grads)

    with open(args.out, "w") as f:
        json.dump(
            {
                "args": {**vars(args), "out": str(args.out)},
                "avg_cosine_similarity": avg_cosine_similarity,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
