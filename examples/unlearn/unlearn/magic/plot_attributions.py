#!/usr/bin/env python3
"""Generate MAGIC attribution plots with token-level text highlighting."""

import json
import os
import sys

import matplotlib

matplotlib.rcParams["text.parse_math"] = False
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

output_dir = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/projects/a6a/public/lucia/runs/magic_wmdp_forget_10k_fp32_output"
)
dataset_name = os.path.basename(output_dir.rstrip("/")).replace("_output", "")
eval_name = "MMLU" if dataset_name.endswith("_mmlu") else "WMDP"
# Extract training data name: magic_ultrachat_msl1024_mmlu -> ultrachat
train_data = dataset_name.removeprefix("magic_").removesuffix("_mmlu")
train_data = train_data.split("_msl")[0] if "_msl" in train_data else train_data
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

scores = torch.load(os.path.join(output_dir, "per_token_scores.pt"), weights_only=True)
with open(os.path.join(output_dir, "results.json")) as f:
    results = json.load(f)

example_scores = scores.sum(dim=1)
sorted_idx = example_scores.argsort()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/deep-ignorance-unfiltered")

print(f"Scores: {scores.shape}")

# Shared colormap
cmap = plt.cm.RdBu_r


def plot_token_heatmap(text, tok_scores_1d, title, save_path, max_tokens=256):
    """Render tokens as colored text boxes, background = attribution score."""
    tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    tok_strings = [tokenizer.decode([t]) for t in tokens]
    tok_vals = tok_scores_1d[: len(tokens)]

    vmax = max(np.abs(tok_vals).max(), 1e-12)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

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

    row_height = 0.25
    fig_height = max(3, len(lines) * row_height + 1.2)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(lines))
    ax.invert_yaxis()
    ax.axis("off")

    # Render tokens
    for row_i, line in enumerate(lines):
        x = 0.01
        for tok_i, display in line:
            val = float(tok_vals[tok_i])
            color = cmap(norm(val))
            # Text contrast
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if brightness < 0.5 else "black"

            char_width = 0.0085
            box_width = len(display) * char_width + 0.004

            rect = plt.Rectangle(
                (x, row_i + 0.08), box_width, 0.84, facecolor=color, edgecolor="none"
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

    ax.set_title(title, fontsize=9, loc="left", pad=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label("Token attribution", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 1. Top 10 and bottom 10 bar chart ────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

for ax, label, indices in [
    (
        axes[0],
        f"Top 10 increase {eval_name} loss (most positive attribution)",
        sorted_idx[-10:].flip(0),
    ),
    (
        axes[1],
        f"Top 10 decrease {eval_name} loss (most negative attribution)",
        sorted_idx[:10],
    ),
]:
    vals = example_scores[indices].numpy()
    labels = []
    for idx in indices:
        idx_int = int(idx)
        text = None
        for entry in results.get("highest", []) + results.get("lowest", []):
            if entry["index"] == idx_int:
                text = entry["text"]
                break
        if text is None:
            text = f"(idx {idx_int})"
        text = text[:80].replace("\n", " ").strip()
        labels.append(f"[{idx_int}] {text}")

    colors = ["forestgreen" if v > 0 else "firebrick" for v in vals]
    y_pos = range(len(vals))
    ax.barh(y_pos, vals, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Attribution score (sum over tokens)")
    ax.set_title(label)

plt.tight_layout()
plt.savefig(
    os.path.join(plot_dir, f"{dataset_name}_increase_decrease_loss_10.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print(f"Saved {dataset_name}_increase_decrease_loss_10.png")

# ── 2. Token heatmaps for bottom 10 (most negative) ─────────────
for rank_i in range(10):
    idx_int = int(sorted_idx[rank_i])
    ex_score = float(example_scores[idx_int])
    tok_scores_np = scores[idx_int].numpy()

    text = None
    for entry in results.get("lowest", []):
        if entry["index"] == idx_int:
            text = entry["text"]
            break
    if text is None:
        text = "(text not available)"

    title = f"{train_data} #{rank_i} idx={idx_int}  score={ex_score:.4e}  (blue = decrease {eval_name} loss, red = increase)"
    fname = f"{dataset_name}_tokens_decrease_loss_{rank_i:02d}_idx{idx_int}.png"
    save_path = os.path.join(plot_dir, fname)
    plot_token_heatmap(text, tok_scores_np, title, save_path)
    print(f"Saved {fname}")

# ── 3. Token heatmaps for top 10 (most positive) ────────────────
for rank_i in range(10):
    idx_int = int(sorted_idx[-(rank_i + 1)])
    ex_score = float(example_scores[idx_int])
    tok_scores_np = scores[idx_int].numpy()

    text = None
    for entry in results.get("highest", []):
        if entry["index"] == idx_int:
            text = entry["text"]
            break
    if text is None:
        text = "(text not available)"

    title = f"{train_data} #{rank_i} idx={idx_int}  score={ex_score:.4e}  (blue = decrease {eval_name} loss, red = increase)"
    fname = f"{dataset_name}_tokens_increase_loss_{rank_i:02d}_idx{idx_int}.png"
    save_path = os.path.join(plot_dir, fname)
    plot_token_heatmap(text, tok_scores_np, title, save_path)
    print(f"Saved {fname}")

print(f"\nAll plots saved to {plot_dir}")
