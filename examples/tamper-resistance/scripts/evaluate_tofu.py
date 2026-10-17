#!/usr/bin/env python3
"""Evaluate the ROUGE-L metrics used by SSIUU Table 2."""

import argparse
import json
from pathlib import Path
from statistics import mean


def rouge_l_recall(reference: str, prediction: str) -> float:
    """Match TOFU's official stemmed ROUGE-L recall metric."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].recall


def generate_predictions(model, tokenizer, questions, batch_size=8):
    """Generate deterministic answers in left-padded batches."""
    import torch

    predictions = []
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(questions), batch_size):
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for question in questions[start : start + batch_size]
            ]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(
                model.device
            )
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=200,
                    pad_token_id=tokenizer.eos_token_id,
                )
            predictions.extend(
                tokenizer.batch_decode(
                    generated[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            )
    finally:
        tokenizer.padding_side = padding_side
    return predictions


def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def forget_score(model_path, batch_size=8):
    from datasets import load_dataset

    dataset = load_dataset("locuslab/TOFU", "forget01", split="train")
    model, tokenizer = load_model(model_path)
    predictions = generate_predictions(
        model, tokenizer, dataset["question"], batch_size=batch_size
    )
    scores = [
        rouge_l_recall(answer, prediction)
        for answer, prediction in zip(dataset["answer"], predictions)
    ]
    return 100 * mean(scores), predictions


def evaluate(model_path: str, output: Path) -> None:
    from datasets import load_dataset

    model, tokenizer = load_model(model_path)
    scores = {}
    rows = []
    for split in ("forget01", "retain99", "real_authors", "world_facts"):
        dataset = load_dataset("locuslab/TOFU", split, split="train")
        predictions = generate_predictions(model, tokenizer, dataset["question"])
        split_scores = []
        for index, (row, prediction) in enumerate(zip(dataset, predictions)):
            score = rouge_l_recall(row["answer"], prediction)
            split_scores.append(score)
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "prediction": prediction,
                    "rouge_l": score,
                }
            )
        scores[split] = 100 * mean(split_scores)

    summary = {
        "FS": scores["forget01"],
        "RS": scores["retain99"],
        "US": mean((scores["real_authors"], scores["world_facts"])),
        "splits": scores,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.model, args.output)


if __name__ == "__main__":
    main()
