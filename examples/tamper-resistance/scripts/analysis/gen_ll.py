import argparse
import itertools
import os
import os.path as osp
import pandas as pd
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
    parser.add_argument("--radii", type=float, nargs="+", default=[0.05])
    parser.add_argument("--n-pts", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
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
    rows = []

    # gen loss landscape points
    model.eval()
    params = dict(model.named_parameters())
    with torch.no_grad():
        for radius in args.radii:
            coords = torch.linspace(-radius, radius, args.n_pts).tolist()
            for c_i, c_j in itertools.product(coords, coords):
                # add c_i r1 + c_j r2
                for n, p in params.items():
                    p.add_(directions[0][n], alpha=c_i)
                    p.add_(directions[1][n], alpha=c_j)

                loss = 0
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss += model(**batch).loss.item()
                rows.append(
                    {
                        "radius": radius,
                        "alpha": c_i,
                        "beta": c_j,
                        "loss": loss / len(dataloader),
                    }
                )

                # subtract c_i r1 + c_j r2
                for n, p in params.items():
                    p.sub_(directions[0][n], alpha=c_i)
                    p.sub_(directions[1][n], alpha=c_j)

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(osp.join(args.outdir, "grid.csv"), index=False)

    # compute the coords
    rows = []
    for model_name_or_path in args.trajectory:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        p = dict(model.named_parameters())
        z0 = sum(
            [pi.sum() for pi in dict_bin_op(p, directions[0], torch.multiply).values()]
        )
        z1 = sum(
            [pi.sum() for pi in dict_bin_op(p, directions[1], torch.multiply).values()]
        )
        rows.append(
            {"checkpoint": model_name_or_path, "alpha": z0.item(), "beta": z1.item()}
        )

    df = pd.DataFrame(rows)
    df.to_csv(osp.join(args.outdir, "trajectory.csv"), index=False)
