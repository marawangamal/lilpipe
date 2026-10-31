import argparse
import itertools

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_directions(model, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    directions = []
    for _ in range(2):
        d = dict()
        for n, p in model.named_parameters():
            d[n] = torch.randn_like(p)
        directions.append(d)

    # normalize
    # ... normalize so each direction is a unit vector
    return directions


def get_dataloader(dataset_name, tokenizer, max_seq_len, batch_size, limit):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.filter(lambda row: len(row["text"]) > 50)
    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        ),
        batched=True,
        remove_columns=ds.column_names,
    )

    ds = ds.select(range(min(limit, len(ds))))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
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
    args = parser.parse_args()

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    # load data
    dataloader = get_dataloader(
        args.dataset, tok, args.max_seq_len, args.batch_size, args.limit
    )

    # gen loss landscape points
    coords = torch.linspace(-args.radius, args.radius, args.n_pts)
    for batch in dataloader:
        for c_i, c_j in itertools.product(coords, coords):
            p = model.parameters
            for d, c in zip(directions, [c_i, c_j]):
                p = dict_bin_op(p, dict_unary_op(d, lambda t: t * c), torch.sum)
                torch.func.functional_call(model, p, batch)
