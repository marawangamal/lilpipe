"""Plot WMDP learning curves for all active tamper run groups + e2e filter."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/runs")
OUTPUT_PATH = Path(
    "/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/experiment_logs/tamper_wmdp_comparison.png"
)


def parse_tamper_dirname(dirname: str) -> dict | None:
    m = re.match(r"tamper_(.+?)_bio_remove_lr([\d.e-]+)_s(\d+)_(\w+)_(\w+)", dirname)
    if not m:
        return None
    return {
        "model_tag": m.group(1),
        "tamper_lr": m.group(2),
        "max_steps": int(m.group(3)),
        "schedule": m.group(4),
        "dtype": m.group(5),
    }


def read_eval_results(eval_dir: Path) -> dict[int, dict]:
    results = {}
    if not eval_dir.exists():
        return results
    for f in eval_dir.glob("step_*.json"):
        try:
            step = int(f.stem.split("_")[1])
            data = json.loads(f.read_text())
            results[step] = data
        except (ValueError, json.JSONDecodeError, IndexError):
            continue
    return results


def get_best_config_per_group() -> dict[str, tuple[list[int], list[float], str]]:
    """For each model group, find the tamper config with best peak WMDP.
    Returns {model_tag: (steps, wmdp_accs, config_label)}."""
    groups: dict[str, list[tuple[str, dict[int, dict]]]] = {}

    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("tamper_"):
            continue
        info = parse_tamper_dirname(d.name)
        if info is None:
            continue
        eval_dir = d / "eval_results"
        results = read_eval_results(eval_dir)
        if not results:
            continue
        config_label = f"{info['schedule']}/{info['dtype']}/lr{info['tamper_lr']}"
        groups.setdefault(info["model_tag"], []).append((config_label, results))

    best_per_group = {}
    for model_tag, configs in groups.items():
        best_peak = -1
        best_data = None
        best_label = ""
        for label, results in configs:
            peak = max((r.get("wmdp_bio_acc", 0) for r in results.values()), default=0)
            if peak > best_peak:
                best_peak = peak
                best_data = results
                best_label = label

        if best_data:
            steps_sorted = sorted(best_data.keys())
            accs = [best_data[s].get("wmdp_bio_acc", 0) * 100 for s in steps_sorted]
            best_per_group[model_tag] = (steps_sorted, accs, best_label)

    return best_per_group


LOG_PATH = Path(
    "/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/experiment_logs/unrestrained_SFT.md"
)


def load_filter_bio_forget_tamper(metric: str = "WMDP") -> tuple[list[int], list[float]]:
    """Parse lin/wu0/lr2e-5/fp16 bio_forget WMDP or MMLU row from unrestrained_SFT.md."""
    content = LOG_PATH.read_text()
    lines = content.split("\n")

    # Find the header row for the e2e filter tamper table (contains many Step columns)
    header_steps = []
    header_line_idx = None
    for i, line in enumerate(lines):
        if "Step 0" in line and "Step 10000" in line:
            for cell in line.split("|"):
                m = re.match(r"\s*Step (\d+)\s*", cell)
                if m:
                    header_steps.append(int(m.group(1)))
            header_line_idx = i
            break

    if not header_steps or header_line_idx is None:
        return [], []

    # The MMLU row is a continuation row without config/data text.
    # Track when we find the config row, then match the metric.
    found_config_row = False
    for line in lines[header_line_idx + 1 :]:
        if "lin/wu0/lr2e-5/fp16" in line and "bio_forget" in line:
            found_config_row = True
            if metric not in line:
                continue
        if not found_config_row:
            continue
        if metric not in line:
            continue
        # Row: | config | data | metric | steps | Step 0 | Step 500 | ...
        # Step values start at pipe-delimited column index 4 (0-based after split)
        # split("|") gives ["", " config ", " data ", " metric ", " steps ", " val0 ", ...]
        cells = line.split("|")
        # Step value columns start at index 5 in the split result (after leading empty + 4 header cols)
        step_cells = cells[5:]
        step_vals = []
        for c in step_cells:
            cleaned = c.replace("**", "").strip()
            try:
                step_vals.append(float(cleaned))
            except ValueError:
                step_vals.append(None)

        paired = [
            (s, v) for s, v in zip(header_steps, step_vals) if v is not None
        ]
        return [p[0] for p in paired], [p[1] for p in paired]

    return [], []


# To exclude a run, comment out or remove its entry.
# Colors are fixed per model tag so WMDP and MMLU plots are consistent.
_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20))
DISPLAY_NAMES = {
    "cb_sft_ret0_rm10_orth5_lr1e-4": "CB ret0 rm10 orth5 lr1e-4",
    "cb_sft_ret2_rm10_orth5_lr1e-4": "CB ret2 rm10 orth5 lr1e-4",
    "ct_sft_muon_ret0_rm2000": "CT Muon ret0 rm2000",
    "lens_sft_ret0": "Lens ret0 rm5 lr1e-3",
    "lens_sft_ret0_rm100_lr1e-4": "Lens ret0 rm100 lr1e-4",
    "lens_sft_ret5_rm5_lr1e-3": "Lens ret5 lr1e-3",
    "mu_sft_ret1e-3_up1_lr5e-5": "MU ret1e-3 up1 lr5e-5",
    "mu_sft_ret140_up1_lr5e-5": "MU ret140 up1 lr5e-5",
    "seq_sft_ret0_rm5_lr2e-4_nn2": "Seq ret0 rm5 lr2e-4",
    "seq_sft_ret2_rm5_lr2e-4_nn2": "Seq ret2 rm5 lr2e-4",
}
MODEL_COLORS = {tag: _TAB20[i % len(_TAB20)] for i, tag in enumerate(DISPLAY_NAMES)}


def main():
    best_per_group = get_best_config_per_group()
    bio_forget_steps, bio_forget_wmdp = load_filter_bio_forget_tamper()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort by starting WMDP (ascending) for visual clarity
    sorted_tags = sorted(
        best_per_group.keys(),
        key=lambda t: best_per_group[t][1][0] if best_per_group[t][1] else 0,
    )

    for tag in sorted_tags:
        if tag not in DISPLAY_NAMES:
            continue
        steps, accs, config_label = best_per_group[tag]
        name = DISPLAY_NAMES[tag]

        # Subsample if too many points (eval_every=10 runs)
        if len(steps) > 50:
            indices = [0]
            for i in range(1, len(steps)):
                if steps[i] - steps[indices[-1]] >= 450:
                    indices.append(i)
            if indices[-1] != len(steps) - 1:
                indices.append(len(steps) - 1)
            steps = [steps[i] for i in indices]
            accs = [accs[i] for i in indices]

        ax.plot(
            steps,
            accs,
            color=MODEL_COLORS[tag],
            linewidth=2,
            alpha=0.85,
            marker="o",
            markersize=4,
            label=f"{name} ({config_label})",
        )

    # Plot bio_forget filter tamper
    if bio_forget_steps and bio_forget_wmdp:
        ax.plot(
            bio_forget_steps,
            bio_forget_wmdp,
            color="#984ea3",
            linewidth=2.5,
            alpha=0.9,
            marker="D",
            markersize=4,
            linestyle="--",
            label="E2E Filter bio_forget (lin/wu0/lr2e-5/fp16)",
        )

    # Reference lines
    ax.axhline(
        y=42.97,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline WMDP (42.97%)",
    )
    ax.axhline(
        y=25, color="green", linestyle=":", linewidth=1, alpha=0.5, label="Random (25%)"
    )

    ax.set_xlabel("Tampering Step", fontsize=12)
    ax.set_ylabel("WMDP Bio Robust Accuracy (%)", fontsize=12)
    ax.set_title("WMDP Recovery Under bio_remove Tampering: Best Config per Model", fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(15, 50)
    ax.set_xlim(-100, 11000)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
