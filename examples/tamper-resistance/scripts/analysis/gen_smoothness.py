"""Measure WMDP forget-loss landscapes as in Fan et al. (2025), Figure 2."""

import argparse
import json
import os
import os.path as osp
from itertools import product

import numpy as np

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


def direction_gram(directions):
    """Return the 2x2 Gram matrix of the unscaled Gaussian directions."""
    import torch

    gram = np.zeros((2, 2))
    with torch.inference_mode():
        for u, v in zip(*directions, strict=True):
            u = u.to(dtype=torch.float64)
            v = v.to(device=u.device, dtype=torch.float64)
            gram[0, 0] += (u * u).sum().item()
            gram[0, 1] += (u * v).sum().item()
            gram[1, 1] += (v * v).sum().item()
    gram[1, 0] = gram[0, 1]
    return gram


def project_weights(model, reference, directions, gram):
    """Fit the centered weight difference to the unscaled two-direction basis."""
    import torch

    dot_products = np.zeros(2)
    with torch.inference_mode():
        for parameter, original, u, v in zip(
            model.parameters(), reference, *directions, strict=True
        ):
            if parameter.shape != u.shape or parameter.shape != v.shape:
                raise ValueError("Trajectory parameter shapes do not match directions")
            difference = parameter.to(dtype=torch.float64) - original.to(
                device=parameter.device, dtype=torch.float64
            )
            for axis, direction in enumerate((u, v)):
                direction = direction.to(device=parameter.device, dtype=torch.float64)
                dot_products[axis] += (difference * direction).sum().item()
    return np.linalg.solve(gram, dot_products)


def loss_surfaces(
    model, batches, coordinates, seed, direction_stds, *, directions=None
):
    """Evaluate loss(theta + x r1 + y r2) on a square grid for each direction scale.

    The Gaussian directions are sampled once per model and rescaled for every
    standard deviation, so the surfaces differ only in the perturbation radius.
    """
    import torch
    from tqdm.auto import tqdm

    parameters = list(model.parameters())
    center = [parameter.detach().clone() for parameter in parameters]
    if directions is None:
        directions = random_directions(model, seed, 1.0)
    x_direction, y_direction = directions
    torch.cuda.empty_cache()
    surfaces = {}

    for direction_std in direction_stds:
        surface = np.empty((len(coordinates), len(coordinates)))
        points = product(enumerate(coordinates), enumerate(coordinates))
        progress = tqdm(
            points,
            total=surface.size,
            desc=f"Loss surface σ={direction_std:g}",
            unit="point",
        )
        for (row, y), (column, x) in progress:
            with torch.no_grad():
                for parameter, original, x_noise, y_noise in zip(
                    parameters, center, x_direction, y_direction
                ):
                    parameter.copy_(original)
                    parameter.add_(x_noise, alpha=float(x * direction_std))
                    parameter.add_(y_noise, alpha=float(y * direction_std))
            surface[row, column] = prediction_loss(model, batches)
            progress.set_postfix(loss=f"{surface[row, column]:.4f}")
        surfaces[direction_std] = surface

    with torch.no_grad():
        for parameter, original in zip(parameters, center):
            parameter.copy_(original)

    return surfaces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/smoothness/smoothness.json")
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument(
        "--direction-std",
        type=float,
        nargs="+",
        default=[
            1e-3,
            5e-3,
            1e-2,
            2e-2,
            3e-2,
            4e-2,
            5e-2,
            6e-2,
            7e-2,
            8e-2,
            9e-2,
            1e-1,
        ],
        help="Direction scales to evaluate; directions are shared across scales",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--tokenizer", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument(
        "--model",
        required=True,
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Measure one checkpoint; provide exactly one LABEL PATH pair",
    )
    parser.add_argument(
        "--trajectory",
        nargs="+",
        metavar="PATH",
        help="Ordered checkpoints to project relative to --model into each scaled landscape",
    )
    args = parser.parse_args()
    label, path = args.model
    if args.trajectory and any(std <= 0 for std in args.direction_std):
        parser.error("--trajectory requires strictly positive --direction-std values")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    documents = load_dataset(dataset_name, split="train")
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
        "seed": args.seed,
        "models": [],
    }
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)

    print(f"{label}: {path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=getattr(torch, args.dtype),
        device_map="balanced",
        attn_implementation="eager",
    ).eval()
    model.config.use_cache = False
    parameter_layout = [(name, p.shape) for name, p in model.named_parameters()]
    directions = random_directions(model, args.seed, 1.0)
    surfaces = loss_surfaces(
        model,
        batches,
        coordinates,
        args.seed,
        args.direction_std,
        directions=directions,
    )
    for direction_std, surface in surfaces.items():
        results["models"].append(
            {
                "path": path,
                "label": label,
                "direction_std": direction_std,
                "loss": surface.tolist(),
            }
        )
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    reference = [p.detach() for p in model.parameters()] if args.trajectory else None
    del model
    torch.cuda.empty_cache()

    if args.trajectory:
        results["trajectory_projection"] = {
            "type": "centered_least_squares",
            "reference_path": path,
        }
        gram = direction_gram(directions)
        for surface in results["models"]:
            surface["trajectory"] = []
        for checkpoint in args.trajectory:
            print(f"Projecting: {checkpoint}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=getattr(torch, args.dtype),
                device_map="balanced",
                attn_implementation="eager",
            ).eval()
            if [
                (name, p.shape) for name, p in model.named_parameters()
            ] != parameter_layout:
                raise ValueError(f"Trajectory parameter layout differs: {checkpoint}")
            coordinates = project_weights(model, reference, directions, gram)
            for surface in results["models"]:
                surface["trajectory"].append(
                    {
                        "path": checkpoint,
                        "coordinates": (
                            coordinates / surface["direction_std"]
                        ).tolist(),
                    }
                )
            del model
            torch.cuda.empty_cache()
            with open(args.output, "w") as handle:
                json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
