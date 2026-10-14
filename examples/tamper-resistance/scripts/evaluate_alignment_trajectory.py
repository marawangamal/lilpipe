"""Measure held-out harmful-response rate along a relearning trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_alignment import (
    DATASET,
    MODERATION_THRESHOLD,
    MODERATOR,
    REVISION,
    evaluation_rows,
    generate,
    harmful_rate,
    load_manifest,
    moderate,
)

DEFAULT_STEPS = (0, 50, 100, 200, 300, 400, 500)


def checkpoint_paths(aligned_adapter, trajectory_dir, steps):
    """Map requested attack steps to the adapters that represent them."""
    paths = {}
    for step in steps:
        path = (
            Path(aligned_adapter)
            if step == 0
            else Path(trajectory_dir) / f"checkpoint-{step}"
        )
        if not path.is_dir():
            raise FileNotFoundError(f"adapter for attack step {step} not found: {path}")
        paths[step] = path
    return paths


def write_summary(output, rows):
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "step", "harmful_rate", "harmful_percent"),
        )
        writer.writeheader()
        writer.writerows(rows)
    markdown = "| Method | Attack step | Held-out harmful rate |\n|---|---:|---:|\n"
    markdown += "".join(
        f"| {row['method']} | {row['step']} | {row['harmful_rate']:.4f} |\n"
        for row in rows
    )
    (output / "trajectory.md").write_text(markdown)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [row["step"] for row in rows],
        [row["harmful_percent"] for row in rows],
        marker="o",
    )
    axis.set(
        xlabel="Harmful-only relearning step",
        ylabel="Held-out harmful responses (%)",
        ylim=(0, 100),
    )
    figure.tight_layout()
    figure.savefig(output / "heldout-harmful-score.png", dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("method")
    parser.add_argument("base_model")
    parser.add_argument("aligned_adapter")
    parser.add_argument("trajectory_dir")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args(argv)

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    from data.beavertail_alignment import MANIFEST_PATH, split_source_indices

    steps = tuple(dict.fromkeys(args.steps))
    adapters = checkpoint_paths(args.aligned_adapter, args.trajectory_dir, steps)
    dataset = load_dataset(DATASET, split="train", revision=REVISION)
    manifest = load_manifest(MANIFEST_PATH)
    expected = split_source_indices([dataset[index] for index in range(len(dataset))])
    if manifest != expected:
        raise RuntimeError("evaluation split does not match the pinned dataset")
    source_rows = evaluation_rows(dataset, manifest)
    prompts = [row["prompt"] for row in source_rows]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    output = args.result_root / args.model_id
    output.mkdir(parents=True, exist_ok=True)

    for step, adapter in adapters.items():
        generation_path = output / f"generations-step-{step}.json"
        if generation_path.exists():
            print(f"reusing generations for step {step}", flush=True)
            continue
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, adapter)
        generations = generate(model, tokenizer, prompts)
        records = [
            {
                "source_index": row["source_index"],
                "prompt": prompt,
                "generation": generation,
            }
            for row, prompt, generation in zip(
                source_rows, prompts, generations, strict=True
            )
        ]
        generation_path.write_text(json.dumps(records, indent=2) + "\n")
        print(f"saved generations for step {step}", flush=True)
        del model, base
        torch.cuda.empty_cache()

    moderator_tokenizer = AutoTokenizer.from_pretrained(MODERATOR)
    moderator = AutoModelForSequenceClassification.from_pretrained(
        MODERATOR, torch_dtype=torch.bfloat16, device_map="auto"
    )
    trajectory = []
    for step in steps:
        generation_path = output / f"generations-step-{step}.json"
        records = json.loads(generation_path.read_text())
        generations = [record["generation"] for record in records]
        scores = moderate(moderator_tokenizer, moderator, prompts, generations)
        for record, score in zip(records, scores, strict=True):
            record["harmful_score"] = score
        generation_path.write_text(json.dumps(records, indent=2) + "\n")
        rate = harmful_rate(scores)
        trajectory.append(
            {
                "method": args.method,
                "step": step,
                "harmful_rate": rate,
                "harmful_percent": 100 * rate,
            }
        )
        write_summary(output, trajectory)
        print(f"step {step}: held-out harmful rate {rate:.4f}", flush=True)


if __name__ == "__main__":
    main()
