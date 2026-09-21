#!/usr/bin/env python3
"""Find sequences that most increase WMDP loss while decreasing MMLU loss.

For each training dataset, loads both WMDP and MMLU per-token scores,
ranks examples by (WMDP_score - MMLU_score), and generates heatmaps
showing both WMDP and MMLU token attributions side by side.
"""

import os
import sys

import matplotlib

matplotlib.rcParams["text.parse_math"] = False
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

base_dir = sys.argv[1] if len(sys.argv) > 1 else "/projects/a6a/public/lucia/runs"
datasets = (
    sys.argv[2:]
    if len(sys.argv) > 2
    else ["ultrachat", "wikitext", "wmdp_lie_o", "wmdp_retain"]
)


def load_train_dataset(name, max_seq_len=1024):
    """Load and chunk training dataset, matching magic_per_token.py logic."""
    if name == "wmdp_forget":
        ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    elif name == "wmdp_retain":
        ds = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
    elif name == "wmdp_lie_o":
        ds = load_dataset("Unlearning/wmdp-lie-o-deep-fried", split="train")
    elif name == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        ds = ds.map(lambda x: {"text": "\n".join(m["content"] for m in x["messages"])})
    elif name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    ds = ds.filter(lambda x: len(x["text"].strip()) > 100)

    # Chunk documents into max_seq_len-token pieces
    tok = AutoTokenizer.from_pretrained("EleutherAI/deep-ignorance-unfiltered")
    all_ids = []
    for example in ds:
        all_ids.extend(tok.encode(example["text"], add_special_tokens=False))
    n_chunks = len(all_ids) // max_seq_len
    chunks = []
    for i in range(n_chunks):
        chunk_ids = all_ids[i * max_seq_len : (i + 1) * max_seq_len]
        chunks.append(tok.decode(chunk_ids))
    return chunks


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/deep-ignorance-unfiltered")
cmap = plt.cm.RdBu_r


def plot_dual_heatmap(text, wmdp_scores, mmlu_scores, title, save_path, max_tokens=256):
    """Two rows of token heatmaps: WMDP on top, MMLU on bottom."""
    tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    tok_strings = [tokenizer.decode([t]) for t in tokens]
    wmdp_vals = wmdp_scores[: len(tokens)]
    mmlu_vals = mmlu_scores[: len(tokens)]

    # Layout: wrap tokens into lines
    chars_per_line = 100
    lines = []
    current_line = []
    current_len = 0
    for i, ts in enumerate(tok_strings):
        display = ts.replace("\n", "\\n")
        if current_len + len(display) > chars_per_line and current_line:
            lines.append(current_line)
            current_line = []
            current_len = 0
        current_line.append((i, display))
        current_len += len(display)
    if current_line:
        lines.append(current_line)

    n_lines = len(lines)
    row_height = 0.25
    gap = 0.6
    fig_height = max(4, 2 * n_lines * row_height + gap + 2.0)
    fig, (ax_wmdp, ax_mmlu) = plt.subplots(
        2, 1, figsize=(18, fig_height), gridspec_kw={"hspace": 0.4}
    )

    for ax, vals, label in [(ax_wmdp, wmdp_vals, "WMDP"), (ax_mmlu, mmlu_vals, "MMLU")]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_lines)
        ax.invert_yaxis()
        ax.axis("off")

        vmax = max(np.abs(vals).max(), 1e-12)
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

        for row_i, line in enumerate(lines):
            x = 0.01
            for tok_i, display in line:
                val = float(vals[tok_i])
                color = cmap(norm(val))
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                text_color = "white" if brightness < 0.5 else "black"

                char_width = 0.0085
                box_width = len(display) * char_width + 0.004

                rect = plt.Rectangle(
                    (x, row_i + 0.08),
                    box_width,
                    0.84,
                    facecolor=color,
                    edgecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x + 0.002,
                    row_i + 0.5,
                    display,
                    fontsize=7,
                    fontfamily="monospace",
                    color=text_color,
                    verticalalignment="center",
                )
                x += box_width + 0.001

        score_sum = float(vals.sum())
        ax.set_title(
            f"{label} tokens (sum={score_sum:.4e}, blue = decrease loss, red = increase)",
            fontsize=9,
            loc="left",
            pad=8,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=30)
        cbar.set_label(f"{label} token attribution", fontsize=8)

    fig.suptitle(title, fontsize=10, y=1.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


for ds in datasets:
    wmdp_dir = os.path.join(base_dir, f"magic_{ds}_msl1024_output")
    mmlu_dir = os.path.join(base_dir, f"magic_{ds}_msl1024_mmlu_output")

    if not os.path.isdir(wmdp_dir) or not os.path.isdir(mmlu_dir):
        print(f"Skipping {ds}: missing WMDP or MMLU output dir")
        continue

    print(f"=== {ds} ===")
    wmdp_scores = torch.load(
        os.path.join(wmdp_dir, "per_token_scores.pt"), weights_only=True
    )
    mmlu_scores = torch.load(
        os.path.join(mmlu_dir, "per_token_scores.pt"), weights_only=True
    )

    print(f"Loading {ds} training data...")
    train_texts = load_train_dataset(ds)
    n_examples = wmdp_scores.shape[0]
    print(f"Loaded {len(train_texts)} chunks, using first {n_examples}")

    wmdp_example = wmdp_scores.sum(dim=1)
    mmlu_example = mmlu_scores.sum(dim=1)

    # Selective score: increases WMDP loss (positive) while decreasing MMLU loss (negative)
    selective = wmdp_example - mmlu_example
    selective_idx = selective.argsort(descending=True)

    plot_dir = os.path.join(base_dir, f"magic_{ds}_msl1024_selective_output", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    top_n = 10
    indices = selective_idx[:top_n]
    vals_wmdp = wmdp_example[indices].numpy()
    vals_mmlu = mmlu_example[indices].numpy()

    labels = []
    for idx in indices:
        idx_int = int(idx)
        text = (
            train_texts[idx_int] if idx_int < len(train_texts) else f"(idx {idx_int})"
        )
        text = text[:70].replace("\n", " ").strip()
        labels.append(f"[{idx_int}] {text}")

    y_pos = np.arange(top_n)
    ax.barh(
        y_pos - 0.15,
        vals_wmdp,
        height=0.3,
        color="firebrick",
        label="WMDP score (want positive)",
    )
    ax.barh(
        y_pos + 0.15,
        vals_mmlu,
        height=0.3,
        color="steelblue",
        label="MMLU score (want negative)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Attribution score (sum over tokens)")
    ax.set_title(f"{ds}: top 10 selective (increase WMDP loss, decrease MMLU loss)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"magic_{ds}_msl1024_selective_top10.png"
    plt.savefig(os.path.join(plot_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

    # Dual heatmaps for top 10 selective
    for rank_i in range(10):
        idx_int = int(selective_idx[rank_i])
        wmdp_s = float(wmdp_example[idx_int])
        mmlu_s = float(mmlu_example[idx_int])
        sel_s = float(selective[idx_int])

        text = (
            train_texts[idx_int]
            if idx_int < len(train_texts)
            else "(text not available)"
        )

        title = (
            f"{ds} selective #{rank_i}  idx={idx_int}  "
            f"WMDP={wmdp_s:.4e}  MMLU={mmlu_s:.4e}  diff={sel_s:.4e}"
        )
        fname = f"magic_{ds}_msl1024_selective_{rank_i:02d}_idx{idx_int}.png"
        save_path = os.path.join(plot_dir, fname)
        plot_dual_heatmap(
            text,
            wmdp_scores[idx_int].numpy(),
            mmlu_scores[idx_int].numpy(),
            title,
            save_path,
        )
        print(f"Saved {fname}")

    print(f"All plots saved to {plot_dir}\n")
