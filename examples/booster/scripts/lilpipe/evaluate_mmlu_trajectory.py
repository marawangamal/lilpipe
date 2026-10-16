#!/usr/bin/env python3
"""Evaluate MMLU along the harmful-only fresh-LoRA attack trajectory."""

import argparse
import json
import os

STEPS = tuple(range(0, 2001, 200))


def adapter_paths(model_root, method, attack, step):
    defense = os.path.join(model_root, f"booster-figure3--defense-{method}")
    if step == 0:
        return defense, None
    attack = os.path.join(
        model_root,
        f"booster-figure3--defense-{method}--attack-{attack}-fresh-lora",
        f"checkpoint-{step}",
    )
    return defense, attack


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("sft", "booster"), required=True)
    parser.add_argument(
        "--attack", choices=("harmful-only", "harmful-100"), default="harmful-only"
    )
    parser.add_argument("--step", choices=STEPS, type=int, required=True)
    parser.add_argument("--model-root", default="artifacts/mila/models")
    parser.add_argument("--output-root", default="artifacts/mila/evals/figure3")
    return parser.parse_args()


def main():
    args = parse_args()
    output = os.path.join(
        args.output_root,
        f"{args.attack}-fresh-lora",
        args.method,
        f"step-{args.step}",
        "mmlu.json",
    )
    if os.path.isfile(output):
        print(f"reusing {output}", flush=True)
        return

    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = "meta-llama/Llama-2-7b-hf"
    defense, attack_adapter = adapter_paths(
        args.model_root, args.method, args.attack, args.step
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, defense)
    if attack_adapter is not None:
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, attack_adapter)
    model.eval()

    results = simple_evaluate(
        model=HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size="auto",
        ),
        tasks=["mmlu"],
        num_fewshot=0,
        batch_size="auto",
    )
    metrics = results["groups"]["mmlu"]
    row = {
        "method": args.method,
        "attack": args.attack,
        "attack_lora_init": "fresh",
        "step": args.step,
        "mmlu_acc": metrics["acc,none"],
        "mmlu_acc_stderr": metrics["acc_stderr,none"],
        "num_fewshot": 0,
        "defense_adapter": defense,
        "attack_adapter": attack_adapter,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = f"{output}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
