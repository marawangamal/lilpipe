"""Print the number and L2 norm of a causal language model's parameters."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sq_norm = sum(p.float().pow(2).sum().item() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}")
    print(f"params: {num_params:,}")
    print(f"param norm: {sq_norm**0.5:.4f}")


if __name__ == "__main__":
    main()
