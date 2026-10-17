#!/usr/bin/env python3
"""Evaluate forget-set ROUGE after one Table 2 retraining attack."""

import argparse
import json
from pathlib import Path
from statistics import mean

from evaluate_tofu import generate_predictions, rouge_l_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--attack", choices=("harmful", "benign"), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_dataset("locuslab/TOFU", "forget01", split="train")
    excluded = set()
    if args.attack == "harmful":
        if args.fraction is None:
            parser.error("--fraction is required for a harmful attack")
        from data.tofu import harmful_indices

        excluded = set(harmful_indices(len(rows), args.fraction, args.seed))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    evaluation = [
        (index, row) for index, row in enumerate(rows) if index not in excluded
    ]
    predictions = generate_predictions(
        model, tokenizer, [row["question"] for _, row in evaluation]
    )
    scores = []
    generations = []
    for (index, row), prediction in zip(evaluation, predictions):
        score = rouge_l_recall(row["answer"], prediction)
        scores.append(score)
        generations.append({"index": index, "prediction": prediction, "rouge_l": score})

    result = {
        "attack": args.attack,
        "fraction": args.fraction,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "FS": 100 * mean(scores),
        "evaluation_rows": len(scores),
        "excluded_indices": sorted(excluded),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "generations.jsonl").open("w") as handle:
        for row in generations:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
