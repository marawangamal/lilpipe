"""Measure WMDP forget-loss landscapes as in Fan et al. (2025), Figure 2."""

import argparse
import json
import os
import os.path as osp
from itertools import product

import numpy as np

configs = [
    ("HuggingFaceH4/zephyr-7b-beta", "Origin"),
    ("artifacts/tamia/models/z7b-wmdp-bio--unlearn-fft-npo", "NPO"),
    (
        "artifacts/tamia/models/z7b-wmdp-bio--unlearn-fft-npo-sam-rho1e-3",
        "NPO + SAM",
    ),
]

dataset_name = "cais/wmdp-bio-forget-corpus"


def make_batches(tokenizer, documents, limit, batch_size, sequence_length):
    """Tokenize the forget corpus exactly as in the authors-compatible configs."""
    import torch

    documents = [
        row["text"]
        for row in documents
        if isinstance(row.get("text"), str) and len(row["text"]) > 50
    ][:limit]
    for start in range(0, len(documents), batch_size):
        batch = tokenizer(
            documents[start : start + batch_size],
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # The paper code computes cross entropy on padded EOS positions too.
        batch["labels"] = batch["input_ids"].clone()
        yield {name: value for name, value in batch.items() if torch.is_tensor(value)}


def random_directions(model, seed, std):
    """Sample two Gaussian directions in parameter space."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    directions = [[], []]
    for parameter in model.parameters():
        for direction in directions:
            direction.append(torch.randn_like(parameter) * std)
    return directions


def prediction_loss(model, batches):
    """Return mean next-token cross entropy over the fixed forget documents."""
    import torch

    device = model.get_input_embeddings().weight.device
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in batches:
            batch = {name: value.to(device) for name, value in batch.items()}
            batch_size = batch["input_ids"].shape[0]
            total_loss += model(**batch).loss.item() * batch_size
            total_examples += batch_size
    return total_loss / total_examples


def loss_surface(model, batches, coordinates, seed, direction_std):
    """Evaluate loss(theta + x r1 + y r2) on a square coordinate grid."""
    import torch
    from tqdm.auto import tqdm

    parameters = list(model.parameters())
    center = [parameter.detach().clone() for parameter in parameters]
    x_direction, y_direction = random_directions(model, seed, direction_std)
    torch.cuda.empty_cache()
    surface = np.empty((len(coordinates), len(coordinates)))

    points = product(enumerate(coordinates), enumerate(coordinates))
    progress = tqdm(points, total=surface.size, desc="Loss surface", unit="point")
    for (row, y), (column, x) in progress:
        with torch.no_grad():
            for parameter, original, x_noise, y_noise in zip(
                parameters, center, x_direction, y_direction
            ):
                parameter.copy_(original)
                parameter.add_(x_noise, alpha=float(x))
                parameter.add_(y_noise, alpha=float(y))
        surface[row, column] = prediction_loss(model, batches)
        progress.set_postfix(loss=f"{surface[row, column]:.4f}")

    with torch.no_grad():
        for parameter, original in zip(parameters, center):
            parameter.copy_(original)

    return surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/figures/smoothness.json")
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--direction-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--tokenizer", default=configs[0][0])
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Measure this model; repeat to override the default model set",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    documents = load_dataset(dataset_name, split="train")
    selected_configs = (
        [(path, label) for label, path in args.model] if args.model else configs
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = list(
        make_batches(
            tokenizer,
            documents,
            args.documents,
            args.batch_size,
            args.sequence_length,
        )
    )
    coordinates = np.linspace(-args.radius, args.radius, args.points)
    results = {
        "coordinates": coordinates.tolist(),
        "direction_std": args.direction_std,
        "models": [],
    }
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)

    for path, label in selected_configs:
        print(f"{label}: {path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=getattr(torch, args.dtype),
            device_map="balanced",
            attn_implementation="eager",
        ).eval()
        model.config.use_cache = False
        surface = loss_surface(
            model, batches, coordinates, args.seed, args.direction_std
        )
        results["models"].append(
            {"path": path, "label": label, "loss": surface.tolist()}
        )
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
