"""Plot WMDP forget-loss landscapes as in Fan et al. (2025), Figure 2."""

import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

configs = [
    ("HuggingFaceH4/zephyr-7b-beta", "Origin"),
    ("artifacts/tamia/models/z7b-wmdp-bio--unlearn-fft-npo", "NPO"),
    (
        "artifacts/tamia/models/z7b-wmdp-bio--unlearn-fft-npo-sam-rho1e-3",
        "NPO + SAM",
    ),
]

dataset_name = "cais/wmdp-bio-forget-corpus"
figure_dir = "artifacts/figures"


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


def random_directions(model, seed):
    """Make two Gaussian directions with filter-wise weight normalization."""
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    directions = [[], []]
    for parameter in model.parameters():
        for direction in directions:
            noise = torch.randn_like(parameter)
            if parameter.ndim <= 1:
                noise.zero_()
            else:
                weights = parameter.flatten(1)
                noise = noise.flatten(1)
                noise *= weights.norm(dim=1, keepdim=True) / noise.norm(
                    dim=1, keepdim=True
                ).clamp_min(1e-12)
                noise = noise.reshape_as(parameter)
            direction.append(noise)
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


def loss_surface(model, batches, coordinates, seed):
    """Evaluate loss(theta + x r1 + y r2) on a square coordinate grid."""
    import torch

    parameters = list(model.parameters())
    x_direction, y_direction = random_directions(model, seed)
    surface = np.empty((len(coordinates), len(coordinates)))

    for row, y in enumerate(coordinates):
        for column, x in enumerate(coordinates):
            with torch.no_grad():
                for parameter, x_noise, y_noise in zip(
                    parameters, x_direction, y_direction
                ):
                    parameter.add_(x_noise, alpha=float(x))
                    parameter.add_(y_noise, alpha=float(y))
            try:
                surface[row, column] = prediction_loss(model, batches)
            finally:
                with torch.no_grad():
                    for parameter, x_noise, y_noise in zip(
                        parameters, x_direction, y_direction
                    ):
                        parameter.add_(y_noise, alpha=-float(y))
                        parameter.add_(x_noise, alpha=-float(x))
            print(f"  x={x:+.4f}, y={y:+.4f}, loss={surface[row, column]:.4f}")

    return surface


def plot_surfaces(results, output):
    coordinates = np.asarray(results["coordinates"])
    x, y = np.meshgrid(coordinates, coordinates)
    figure = plt.figure(figsize=(4 * len(results["models"]), 3.5))

    for index, result in enumerate(results["models"], start=1):
        axis = figure.add_subplot(1, len(results["models"]), index, projection="3d")
        axis.plot_surface(x, y, np.asarray(result["loss"]), cmap="viridis")
        axis.set(title=result["label"], xlabel="x", ylabel="y", zlabel="Loss")
        axis.set_zlim(0, 10)

    figure.tight_layout()
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=osp.join(figure_dir, "smoothness.png"))
    parser.add_argument("--data", default=osp.join(figure_dir, "smoothness.json"))
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        help="Plot this model; repeat to override the default model set",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sns.set_theme(style="whitegrid")
    documents = load_dataset(dataset_name, split="train")
    selected_configs = (
        [(path, label) for label, path in args.model] if args.model else configs
    )
    tokenizer = AutoTokenizer.from_pretrained(selected_configs[0][0])
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
    results = {"coordinates": coordinates.tolist(), "models": []}
    os.makedirs(osp.dirname(args.data) or ".", exist_ok=True)

    for path, label in selected_configs:
        print(f"{label}: {path}")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            device_map="balanced",
            attn_implementation="eager",
        ).eval()
        model.config.use_cache = False
        surface = loss_surface(model, batches, coordinates, args.seed)
        results["models"].append(
            {"path": path, "label": label, "loss": surface.tolist()}
        )
        with open(args.data, "w") as handle:
            json.dump(results, handle, indent=2)
        del model
        torch.cuda.empty_cache()

    plot_surfaces(results, args.output)


if __name__ == "__main__":
    main()
