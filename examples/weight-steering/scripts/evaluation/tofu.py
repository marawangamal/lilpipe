#!/usr/bin/env python3
"""Evaluate TOFU forgetting with OpenUnlearning-compatible metric definitions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import harmonic_mean, mean
from typing import Iterable, Sequence

OPENUNLEARNING_REVISION = "4ad738aaf60f6a4385f6e2506d01da99e76c31f3"
TOKEN = re.compile(r"\w+")


def qa_probability(loss: float) -> float:
    """Return the sequence probability used by TOFU from mean answer NLL."""
    return math.exp(-float(loss))


def rouge_l_recall(reference: str, prediction: str) -> float:
    """Compute token-level ROUGE-L recall."""
    reference_tokens = TOKEN.findall(reference.lower())
    prediction_tokens = TOKEN.findall(prediction.lower())
    if not reference_tokens:
        return 0.0
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, 1):
            current.append(
                previous[index - 1] + 1
                if reference_token == prediction_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1] / len(reference_tokens)


def truth_ratio(paraphrased_loss: float, perturbed_losses: Sequence[float]) -> float:
    """Compare paraphrased-answer likelihood with mean perturbed likelihood."""
    if not perturbed_losses:
        raise ValueError("truth_ratio requires at least one perturbed answer")
    return math.exp(float(paraphrased_loss) - mean(map(float, perturbed_losses)))


def normalized_truth_ratio(ratio: float) -> float:
    """Map a truth ratio to the utility score used for non-forget tasks."""
    ratio = float(ratio)
    if ratio < 0:
        raise ValueError("truth ratio cannot be negative")
    return max(0.0, 1.0 - ratio)


def forget_truth_ratio(ratio: float) -> float:
    """Score forgetting by how closely the false/true ratio approaches one."""
    ratio = float(ratio)
    if ratio < 0:
        raise ValueError("truth ratio cannot be negative")
    return min(ratio, 1.0 / (ratio + 1e-10))


def forget_quality(
    truth_ratios: Sequence[float], oracle_ratios: Sequence[float]
) -> float:
    """Return the two-sample KS p-value against the retain-only oracle."""
    if not truth_ratios or not oracle_ratios:
        raise ValueError("forget_quality requires two non-empty distributions")
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(truth_ratios, oracle_ratios).pvalue)
    except ImportError:
        left = sorted(map(float, truth_ratios))
        right = sorted(map(float, oracle_ratios))
        points = sorted(set(left + right))
        distance = max(
            abs(
                sum(value <= point for value in left) / len(left)
                - sum(value <= point for value in right) / len(right)
            )
            for point in points
        )
        effective = len(left) * len(right) / (len(left) + len(right))
        return min(1.0, 2.0 * math.exp(-2.0 * effective * distance * distance))


def model_utility(task_metrics: Iterable[float]) -> float:
    """Return the harmonic mean of positive retain and general-knowledge scores."""
    values = [float(value) for value in task_metrics]
    if not values or any(value <= 0 for value in values):
        return 0.0
    return harmonic_mean(values)


def extraction_strength(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Return the fraction of the answer's longest exactly predicted suffix."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    if not labels:
        return 0.0
    first_exact_suffix = len(labels)
    for index in range(len(labels)):
        if tuple(predictions[index:]) == tuple(labels[index:]):
            first_exact_suffix = index
            break
    return 1.0 - first_exact_suffix / len(labels)


def _answer_loss(model, tokenizer, question: str, answer: str) -> float:
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    encoded = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(model.device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100
    with torch.inference_mode():
        return float(model(input_ids=input_ids, labels=labels).loss)


def _answer_extraction(model, tokenizer, question: str, answer: str) -> float:
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    encoded = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    predictions = logits[:, :-1].argmax(dim=-1)[0, len(prompt_ids) - 1 :].tolist()
    labels = input_ids[0, len(prompt_ids) :].tolist()
    return extraction_strength(predictions, labels)


def _generate(model, tokenizer, question: str) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )


def _rows(config: str):
    from datasets import load_dataset

    evaluation_configs = {
        "forget05": "forget05_perturbed",
        "retain95": "retain_perturbed",
        "real_authors": "real_authors_perturbed",
        "world_facts": "world_facts_perturbed",
    }
    return load_dataset(
        "locuslab/TOFU", evaluation_configs.get(config, config), split="train"
    )


def evaluate_model(
    model_path: str, adapter: str, output: Path, oracle_path: str
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter != "none":
        model = PeftModel.from_pretrained(model, adapter)
    oracle_tokenizer = AutoTokenizer.from_pretrained(
        oracle_path, trust_remote_code=True
    )
    oracle = AutoModelForCausalLM.from_pretrained(
        oracle_path,
        torch_dtype=next(model.parameters()).dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    audits: list[dict[str, object]] = []
    task_scores: dict[str, dict[str, float]] = {}
    oracle_forget_ratios: list[float] = []
    forget_extractions: list[float] = []
    paraphrased_probabilities: list[float] = []
    for task in ("forget05", "retain95", "real_authors", "world_facts"):
        probabilities: list[float] = []
        rouges: list[float] = []
        ratios: list[float] = []
        for index, row in enumerate(_rows(task)):
            question, answer = row["question"], row["answer"]
            paraphrase = row.get("paraphrased_question", question)
            paraphrased_answer = row.get("paraphrased_answer", answer)
            perturbations = row.get("perturbed_answer", [])
            if isinstance(perturbations, str):
                perturbations = [perturbations]
            loss = _answer_loss(model, tokenizer, question, answer)
            paraphrased_loss = _answer_loss(
                model, tokenizer, paraphrase, paraphrased_answer
            )
            perturbed_losses = [
                _answer_loss(model, tokenizer, question, value)
                for value in perturbations
            ]
            ratio = truth_ratio(paraphrased_loss, perturbed_losses)
            generated = _generate(model, tokenizer, question)
            probability = qa_probability(loss)
            rouge = rouge_l_recall(answer, generated)
            probabilities.append(probability)
            rouges.append(rouge)
            ratios.append(ratio)
            audits.append(
                {
                    "task": task,
                    "index": index,
                    "question": question,
                    "reference": answer,
                    "generation": generated,
                    "qa_probability": probability,
                    "rouge_l_recall": rouge,
                    "truth_ratio": ratio,
                }
            )
            if task == "forget05":
                paraphrased_probabilities.append(qa_probability(paraphrased_loss))
                forget_extractions.append(
                    _answer_extraction(model, tokenizer, question, answer)
                )
                oracle_para = _answer_loss(
                    oracle, oracle_tokenizer, paraphrase, paraphrased_answer
                )
                oracle_perturbed = [
                    _answer_loss(oracle, oracle_tokenizer, question, value)
                    for value in perturbations
                ]
                oracle_forget_ratios.append(truth_ratio(oracle_para, oracle_perturbed))
        task_scores[task] = {
            "qa_probability": mean(probabilities),
            "rouge": mean(rouges),
            "truth_ratio": mean(
                (forget_truth_ratio if task == "forget05" else normalized_truth_ratio)(
                    value
                )
                for value in ratios
            ),
        }
        if task == "forget05":
            candidate_forget_ratios = ratios

    paraphrased_score = mean(paraphrased_probabilities)
    utility_inputs = [
        value
        for task in ("retain95", "real_authors", "world_facts")
        for value in task_scores[task].values()
    ]
    summary = {
        "results": {
            "tofu": {
                "forget_quality": forget_quality(
                    candidate_forget_ratios, oracle_forget_ratios
                ),
                "model_utility": model_utility(utility_inputs),
                "qa_probability": task_scores["forget05"]["qa_probability"],
                "rouge": task_scores["forget05"]["rouge"],
                "truth_ratio": task_scores["forget05"]["truth_ratio"],
                "paraphrased_forget": paraphrased_score,
                "extraction_strength": mean(forget_extractions),
            }
        },
        "task_metrics": task_scores,
        "openunlearning_revision": OPENUNLEARNING_REVISION,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "audit.jsonl").open("w") as handle:
        for row in audits:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("adapter")
    parser.add_argument("merge_config")  # registry compatibility; adapter is explicit
    parser.add_argument("output", type=Path)
    parser.add_argument("oracle")
    args = parser.parse_args()
    evaluate_model(args.model, args.adapter, args.output, args.oracle)


if __name__ == "__main__":
    main()
