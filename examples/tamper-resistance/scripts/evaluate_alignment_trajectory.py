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


def heldout_nll(model, tokenizer, rows, batch_size=16, sequence_len=256):
    """Compute token-weighted harmful NLL on held-out prompt/response pairs."""
    import torch

    from data.beavertail_alignment import format_prompt

    features = []
    for row in rows:
        prompt_ids = tokenizer(format_prompt(row["prompt"]), add_special_tokens=True)[
            "input_ids"
        ]
        completion_ids = tokenizer(row["response"].strip(), add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = (prompt_ids + completion_ids + [tokenizer.eos_token_id])[
            :sequence_len
        ]
        prompt_length = min(len(prompt_ids), len(input_ids))
        features.append(
            {
                "input_ids": input_ids,
                "labels": [-100] * prompt_length + input_ids[prompt_length:],
            }
        )

    total_loss = 0.0
    total_tokens = 0
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        width = max(len(feature["input_ids"]) for feature in batch)
        input_ids = torch.full(
            (len(batch), width), pad_token_id, dtype=torch.long, device=model.device
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for index, feature in enumerate(batch):
            length = len(feature["input_ids"])
            input_ids[index, :length] = torch.tensor(
                feature["input_ids"], device=model.device
            )
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(
                feature["labels"], device=model.device
            )
        predicted_tokens = int(labels[:, 1:].ne(-100).sum().item())
        if predicted_tokens == 0:
            raise ValueError("held-out harmful batch contains no completion tokens")
        with torch.inference_mode():
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
        total_loss += float(loss) * predicted_tokens
        total_tokens += predicted_tokens
    return total_loss / total_tokens


def seed_endpoint_generations(endpoint_dir, output, source_rows, final_step):
    """Reuse endpoint generations and scores for trajectory boundary steps."""
    endpoint_path = Path(endpoint_dir) / "generations.json"
    if not endpoint_path.is_file():
        raise FileNotFoundError(f"endpoint generations not found: {endpoint_path}")
    endpoint_records = json.loads(endpoint_path.read_text())
    if len(endpoint_records) != len(source_rows):
        raise ValueError("endpoint generations do not match held-out row count")

    for record, source in zip(endpoint_records, source_rows, strict=True):
        if record["source_index"] != source["source_index"]:
            raise ValueError(
                "endpoint generations do not match held-out source indices"
            )
        if record["prompt"] != source["prompt"]:
            raise ValueError("endpoint generations do not match held-out prompts")

    for step, generation_key, score_key in (
        (0, "before", "before_harmful_score"),
        (final_step, "after", "after_harmful_score"),
    ):
        destination = output / f"generations-step-{step}.json"
        if destination.exists():
            continue
        records = [
            {
                "source_index": record["source_index"],
                "prompt": record["prompt"],
                "generation": record[generation_key],
                "harmful_score": record[score_key],
            }
            for record in endpoint_records
        ]
        destination.write_text(json.dumps(records, indent=2) + "\n")
        print(f"reused endpoint generations and scores for step {step}", flush=True)


def write_summary(output, rows, attack_label="Harmful-only relearning step"):
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "trajectory.csv").open("w", newline="") as handle:
        fieldnames = [
            "method",
            "step",
            "harmful_rate",
            "harmful_percent",
            "heldout_harmful_nll",
        ]
        if "training_harmful_nll" in rows[0]:
            fieldnames.append("training_harmful_nll")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    training_column = (
        " | Training harmful NLL" if "training_harmful_nll" in rows[0] else ""
    )
    markdown = (
        f"| Method | Attack step | Held-out harmful rate | Held-out harmful NLL{training_column} |\n"
        f"|---|---:|---:|---:{'|---:' if training_column else ''}|\n"
    )
    markdown += "".join(
        f"| {row['method']} | {row['step']} | {row['harmful_rate']:.4f} "
        f"| {row['heldout_harmful_nll']:.4f}"
        f"{' | ' + format(row['training_harmful_nll'], '.4f') if training_column else ''} |\n"
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
        xlabel=attack_label,
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
    parser.add_argument("--attack-base-model")
    parser.add_argument("--endpoint-results", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--training-harmful-rows", type=int, default=0)
    parser.add_argument("--attack-label", default="Harmful-only relearning step")
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
    training_rows = [
        dict(dataset[index], source_index=index)
        for index in manifest["relearning"][: args.training_harmful_rows]
    ]
    prompts = [row["prompt"] for row in source_rows]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    output = args.result_root / args.model_id
    output.mkdir(parents=True, exist_ok=True)
    if args.endpoint_results:
        seed_endpoint_generations(
            args.endpoint_results, output, source_rows, max(steps)
        )

    for step, adapter in adapters.items():
        generation_path = output / f"generations-step-{step}.json"
        nll_path = output / f"heldout-nll-step-{step}.json"
        training_nll_path = output / f"training-harmful-nll-step-{step}.json"
        has_training_nll = not training_rows or training_nll_path.exists()
        if generation_path.exists() and nll_path.exists() and has_training_nll:
            print(f"reusing generations and NLL for step {step}", flush=True)
            continue
        base_model_name_or_path = (
            args.base_model if step == 0 else args.attack_base_model or args.base_model
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, adapter)
        if not generation_path.exists():
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
        if not nll_path.exists():
            nll = heldout_nll(model, tokenizer, source_rows)
            nll_path.write_text(json.dumps({"step": step, "nll": nll}) + "\n")
            print(f"step {step}: held-out harmful NLL {nll:.4f}", flush=True)
        if training_rows and not training_nll_path.exists():
            training_nll = heldout_nll(model, tokenizer, training_rows)
            training_nll_path.write_text(
                json.dumps({"step": step, "nll": training_nll}) + "\n"
            )
            print(
                f"step {step}: training harmful NLL {training_nll:.4f}",
                flush=True,
            )
        del model, base
        torch.cuda.empty_cache()

    moderator_tokenizer = None
    moderator = None
    trajectory = []
    for step in steps:
        generation_path = output / f"generations-step-{step}.json"
        records = json.loads(generation_path.read_text())
        generations = [record["generation"] for record in records]
        if all("harmful_score" in record for record in records):
            scores = [record["harmful_score"] for record in records]
            print(f"reusing moderation scores for step {step}", flush=True)
        else:
            if moderator is None:
                moderator_tokenizer = AutoTokenizer.from_pretrained(MODERATOR)
                moderator = AutoModelForSequenceClassification.from_pretrained(
                    MODERATOR, torch_dtype=torch.bfloat16, device_map="auto"
                )
            scores = moderate(moderator_tokenizer, moderator, prompts, generations)
            for record, score in zip(records, scores, strict=True):
                record["harmful_score"] = score
        generation_path.write_text(json.dumps(records, indent=2) + "\n")
        rate = harmful_rate(scores)
        row = {
            "method": args.method,
            "step": step,
            "harmful_rate": rate,
            "harmful_percent": 100 * rate,
            "heldout_harmful_nll": json.loads(
                (output / f"heldout-nll-step-{step}.json").read_text()
            )["nll"],
        }
        if training_rows:
            row["training_harmful_nll"] = json.loads(
                (output / f"training-harmful-nll-step-{step}.json").read_text()
            )["nll"]
        trajectory.append(row)
        write_summary(output, trajectory, args.attack_label)
        print(f"step {step}: held-out harmful rate {rate:.4f}", flush=True)


if __name__ == "__main__":
    main()
