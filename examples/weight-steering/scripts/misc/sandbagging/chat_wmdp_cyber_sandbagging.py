#!/usr/bin/env python
"""Chat interactively with the WMDP-Cyber sandbagging model organism."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen3.8-27B"
ADAPTER = "mremila/Qwen3.8-27B-SFT-WMDP-Cyber-Sandbagging-MO"


def main() -> None:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Model loaded. Press Ctrl+C or Ctrl+D to exit.")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        print(f"\nModel: {response}")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
