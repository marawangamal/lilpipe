import argparse
import os
import os.path as osp

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def title_abstract_text(row, index):
    fields = ("title", "abstract", "text")
    if any(not isinstance(row.get(field), str) for field in fields):
        raise ValueError(f"document {index} lacks title, abstract, or text")
    return "\n\n".join(row[field] for field in fields)


DATASET_TO_PROCESSOR = {
    "cais/wmdp-bio-forget-corpus": title_abstract_text,
}


def main():
    parser = argparse.ArgumentParser(description="Basic causal LM training")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_name_or_path", default=None)
    parser.add_argument("--dataset", default="cais/wmdp-bio-forget-corpus")
    parser.add_argument("--out", default="trained_model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    if args.dataset not in DATASET_TO_PROCESSOR:
        raise ValueError(f"No processor registered for dataset: {args.dataset}")

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_name_or_path,
            is_trainable=True,
        )
    model.to(device)
    model.train()

    dataset = load_dataset(args.dataset, split="train[:1024]")
    process_row = DATASET_TO_PROCESSOR[args.dataset]

    def tokenize(row, index):
        text = process_row(row, index)
        return tokenizer(text, truncation=True, max_length=args.max_length)

    dataset = dataset.map(
        tokenize,
        with_indices=True,
        remove_columns=dataset.column_names,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch = {name: value.to(device) for name, value in batch.items()}

            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

    model.save_pretrained(osp.join(args.out, "model"))
    tokenizer.save_pretrained(osp.join(args.out, "model"))


if __name__ == "__main__":
    main()
