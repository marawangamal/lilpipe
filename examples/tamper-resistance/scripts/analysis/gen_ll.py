import argparse
import itertools
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def dict_unary_op(d, fn):
    return {k: fn(v) for k, v in d.items()}


def dict_bin_op(a, b, fn):
    return {k: fn(a[k], b[k]) for k in a}


def get_directions(model, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    directions = []
    for _ in range(2):
        d = dict()
        for n, p in model.named_parameters():
            d[n] = torch.randn_like(p)
        directions.append(d)

    # normalize so each direction has unit global L2 norm
    normalized = []
    for d in directions:
        norm = sum(t.float().pow(2).sum() for t in d.values()).sqrt()
        normalized.append(dict_unary_op(d, lambda t: t / norm))
    return normalized


def get_dataloader(dataset_name, tokenizer, max_seq_len, batch_size, limit):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.filter(lambda row: len(row["text"]) > 50)
    ds = ds.select(range(min(limit, len(ds))))

    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory", nargs="+")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--n-pts", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # load data
    dataloader = get_dataloader(
        args.dataset, tok, args.max_seq_len, args.batch_size, args.limit
    )

    # get random directions
    directions = get_directions(model, args.seed)
    grid = torch.zeros(args.n_pts, args.n_pts)

    # gen loss landscape points
    model.eval()
    with torch.no_grad():
        coords = torch.linspace(-args.radius, args.radius, args.n_pts)
        for (i, c_i), (j, c_j) in itertools.product(
            enumerate(coords), enumerate(coords)
        ):
            loss = 0
            p = dict(model.named_parameters())
            for d, c in zip(directions, [c_i, c_j]):
                p = dict_bin_op(p, dict_unary_op(d, lambda t: t * c), torch.add)
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = torch.func.functional_call(model, p, args=(), kwargs=batch)
                loss += out.loss.item()
            grid[i, j] = loss / len(dataloader)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(grid, args.out)
