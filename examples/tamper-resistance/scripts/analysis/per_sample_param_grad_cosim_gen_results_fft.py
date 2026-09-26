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


def cosine_similarity(x, y):
    return torch.nn.functional.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0)


def mean(values):
    return torch.stack(tuple(values)).mean()


def dict_binary_op(dict_i, dict_j, op_fn):
    return {key: op_fn(value, dict_j[key]) for key, value in dict_i.items()}


def dict_reduce(values, reduce_fn):
    return reduce_fn(values.values())


def main():
    parser = argparse.ArgumentParser(description="LoRA parameter gradient cosine probe")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_name_or_path")
    parser.add_argument("--dataset", default="cais/wmdp-bio-forget-corpus")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=1024)

    # Fresh LoRA adapter configuration.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"),
    )
    parser.add_argument(
        "--lora_layers_to_transform", nargs="+", type=int, default=tuple(range(31))
    )
    args = parser.parse_args()

    # setup
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(osp.join(args.out, "grads"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if args.adapter_name_or_path:
        # merge the adapter to make a dense model.
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()

    # data
    dataloader = get_dataloader(**vars(args))

    for step, batch in enumerate(dataloader):
        model.zero_grad(set_to_none=True)
        batch = {name: value.to(device) for name, value in batch.items()}
        out = model(**batch)
        out.loss.backward()

        # save grads to disk
        sample_grads = {
            name: param.grad.detach().float().cpu()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        torch.save(sample_grads, osp.join(args.out, "grads", f"sample_{step:05d}.pt"))

    # compute inner products
    del model
    cosims = list()
    for i in range(len(dataloader)):
        for j in range(len(dataloader)):
            gdict_i = torch.load(osp.join(args.out, "grads", f"sample_{i:05d}.pt"))
            gdict_j = torch.load(osp.join(args.out, "grads", f"sample_{i:05d}.pt"))
            cosim = dict_reduce(
                dict_binary_op(gdict_i, gdict_j, op_fn=cosine_similarity),
                reduce_fn=mean,
            )
            cosims.append(cosim)

    with open(args.out, "w") as f:
        json.dump(
            {
                "args": {**vars(args), "out": str(args.out)},
                "avg_cosine_similarity": torch.stack(cosims).mean().item(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
