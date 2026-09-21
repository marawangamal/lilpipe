import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
DATASET_NAME = "Unlearning/WMDP-Bio-Remove-Dataset"
MAX_LENGTH = 1024
BATCH_SIZE = 4
TOP_K_SAVE = 10_000
TOP_K_DISPLAY = 50


def tokenize_example(example, tokenizer):
    text = example["title"] + "\n\n" + example["abstract"] + "\n\n" + example["text"]
    tokens = tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors=None
    )
    return {"input_ids": tokens["input_ids"]}


def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    max_len = max(t.size(0) for t in input_ids)
    padded = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    mask = torch.zeros(len(input_ids), max_len, dtype=torch.long)
    for i, t in enumerate(input_ids):
        padded[i, : t.size(0)] = t
        mask[i, : t.size(0)] = 1
    return {"input_ids": padded, "attention_mask": mask}


def main():
    hf_token = os.environ.get("HF_TOKEN")

    run_path = Path("runs/surprising_tokens")
    run_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    ds = load_dataset(DATASET_NAME, token=hf_token, split="train")
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    actual_vocab = model.config.vocab_size
    print(f"Model vocab size: {actual_vocab}")

    token_count = torch.zeros(actual_vocab, dtype=torch.float64)
    prob_sum = torch.zeros(actual_vocab, dtype=torch.float64)

    # Also track per-occurrence context: store up to a few examples per token
    # for the top surprising ones (collected after the main pass)

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Computing token surprisal"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Shift: predict position t+1 from position t
            shift_logits = logits[:, :-1, :]  # (B, T-1, V)
            shift_labels = input_ids[:, 1:]  # (B, T-1)
            shift_mask = attention_mask[:, 1:]  # (B, T-1)

            # Get probabilities for all tokens
            probs = torch.softmax(shift_logits.float(), dim=-1)  # (B, T-1, V)

            # Gather probability assigned to the actual next token
            label_probs = probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (B, T-1)

            # Mask out padding
            label_probs = label_probs * shift_mask.float()

            # Accumulate into per-token stats
            for b in range(input_ids.size(0)):
                valid_len = shift_mask[b].sum().item()
                if valid_len == 0:
                    continue
                labels_b = shift_labels[b, : int(valid_len)].cpu()
                probs_b = label_probs[b, : int(valid_len)].cpu().to(torch.float64)

                token_count.scatter_add_(0, labels_b, torch.ones_like(probs_b))
                prob_sum.scatter_add_(0, labels_b, probs_b)

                total_tokens += int(valid_len)

    print(f"Total tokens processed: {total_tokens}")

    # Compute mean probability per token (only for tokens that appeared)
    appeared_mask = token_count > 0
    mean_prob = torch.zeros_like(prob_sum)
    mean_prob[appeared_mask] = prob_sum[appeared_mask] / token_count[appeared_mask]

    # For tokens that never appeared, set mean_prob to 1.0 so they sort last
    mean_prob[~appeared_mask] = 1.0

    # Sort by mean probability (ascending = most surprising first)
    sorted_indices = mean_prob.argsort()

    # Only keep tokens that actually appeared
    sorted_indices_appeared = sorted_indices[appeared_mask[sorted_indices]]

    top_k_ids = sorted_indices_appeared[:TOP_K_SAVE].tolist()
    top_k_mean_probs = mean_prob[top_k_ids].tolist()
    top_k_counts = token_count[top_k_ids].tolist()

    results = []
    for rank, token_id in enumerate(top_k_ids):
        token_str = tokenizer.decode([token_id])
        results.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token_str": token_str,
                "mean_prob": top_k_mean_probs[rank],
                "count": int(top_k_counts[rank]),
            }
        )

    with open(run_path / "top_10k_surprising_tokens.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save raw arrays for later use
    torch.save(
        {
            "token_count": token_count,
            "prob_sum": prob_sum,
            "mean_prob": mean_prob,
        },
        run_path / "token_stats.pt",
    )

    print(
        f"\nSaved top {TOP_K_SAVE} tokens to "
        f"{run_path / 'top_10k_surprising_tokens.json'}"
    )
    print(f"Saved raw stats to {run_path / 'token_stats.pt'}")

    # Now collect in-context examples for top 50
    print(f"\nCollecting context for top {TOP_K_DISPLAY} most surprising tokens...")
    top_display_ids = set(top_k_ids[:TOP_K_DISPLAY])

    # Second pass: collect context snippets
    context_examples = {tid: [] for tid in top_display_ids}
    MAX_EXAMPLES_PER_TOKEN = 3

    dl2 = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    done = False
    with torch.no_grad():
        for batch in tqdm(dl2, desc="Collecting context examples"):
            if done:
                break
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            probs = torch.softmax(shift_logits.float(), dim=-1)
            top_pred_ids = probs.argmax(dim=-1)  # (B, T-1) model's top prediction

            for b in range(input_ids.size(0)):
                valid_len = int(shift_mask[b].sum().item())
                if valid_len == 0:
                    continue
                for t in range(valid_len):
                    tid = shift_labels[b, t].item()
                    if (
                        tid in top_display_ids
                        and len(context_examples[tid]) < MAX_EXAMPLES_PER_TOKEN
                    ):
                        # Context: 20 tokens before, the target token
                        start = max(0, t + 1 - 20)
                        end = t + 2  # include the target token
                        context_ids = input_ids[b, start:end].cpu().tolist()
                        context_str = tokenizer.decode(context_ids)
                        predicted_id = top_pred_ids[b, t].item()
                        predicted_str = tokenizer.decode([predicted_id])
                        prob_val = probs[b, t, tid].item()

                        context_examples[tid].append(
                            {
                                "context": context_str,
                                "target_token": tokenizer.decode([tid]),
                                "model_predicted": predicted_str,
                                "prob_assigned": prob_val,
                            }
                        )

            # Check if we have enough examples for all tokens
            if all(len(v) >= MAX_EXAMPLES_PER_TOKEN for v in context_examples.values()):
                done = True

    # Print top 50
    print(f"\n{'='*100}")
    print(f"Top {TOP_K_DISPLAY} most surprising tokens in the bio unlearning dataset")
    print(f"{'='*100}\n")

    for entry in results[:TOP_K_DISPLAY]:
        tid = entry["token_id"]
        print(
            f"Rank {entry['rank']:3d} | Token: {repr(entry['token_str']):30s} | "
            f"Mean P: {entry['mean_prob']:.6f} | Count: {entry['count']}"
        )
        examples = context_examples.get(tid, [])
        for ex in examples:
            ctx = ex["context"].replace("\n", "\\n")
            if len(ctx) > 120:
                ctx = "..." + ctx[-117:]
            print(f"         Context: {ctx}")
            print(
                f"         Model predicted: {repr(ex['model_predicted'])} | "
                f"P(actual): {ex['prob_assigned']:.6f}"
            )
        print()

    # Save context examples too
    serializable_contexts = {}
    for tid, exs in context_examples.items():
        serializable_contexts[str(tid)] = exs

    with open(run_path / "top50_with_context.json", "w") as f:
        json.dump(
            {
                "tokens": results[:TOP_K_DISPLAY],
                "contexts": serializable_contexts,
            },
            f,
            indent=2,
        )

    print(f"Saved context examples to {run_path / 'top50_with_context.json'}")


if __name__ == "__main__":
    main()
