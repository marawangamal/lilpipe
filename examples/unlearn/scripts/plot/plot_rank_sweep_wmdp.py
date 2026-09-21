#!/usr/bin/env python3
"""Plot WMDP recovery curves for rank sweep tamper attacks + filtered model.

For each rank, selects the best-performing tamper config (highest peak WMDP)
across all available sweep configs. Shows both old (multi-epoch) and new (1-epoch)
data where available.
"""

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(pattern):
    """Load the most recent results JSON matching the glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def load_results_from_log(log_path):
    """Parse eval results from a SLURM .out log file (for in-progress runs)."""
    results = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Step (\d+): WMDP Bio Acc = ([\d.]+)", line)
            if m:
                entry = {"step": int(m.group(1)), "wmdp_bio_acc": float(m.group(2))}
                mmlu = re.search(r"MMLU = ([\d.]+)", line)
                if mmlu:
                    entry["mmlu_acc"] = float(mmlu.group(1))
                results.append(entry)
    return results if results else None


def find_best_config(runs_dir, rank):
    """Find the tamper config with highest peak WMDP for a given rank.

    Checks: old multi-epoch dirs,
    rank sweep 1ep dirs/logs,
    ret coef 1ep dirs/logs (r=32 only).
    """
    sweep_tags = [
        "cosine_wr01_lr2e-5",
        "cosine_wr01_lr2e-4",
        "cosine_wr01_lr1e-3",
        "cosine_ws30_lr2e-4",
        "cosine_ws100_lr2e-4",
        "constant_wr01_lr2e-4",
        "cosine_warmup_lr2e-4",
    ]
    sweep_short_map = {
        "cosine_wr01_lr2e-5": "lr5",
        "cosine_wr01_lr2e-4": "std",
        "cosine_wr01_lr1e-3": "lr3",
        "cosine_ws30_lr2e-4": "ws30",
        "cosine_ws100_lr2e-4": "ws100",
        "constant_wr01_lr2e-4": "con",
    }

    best_data = None
    best_peak = -1
    best_name = None
    best_is_1ep = False

    # 1ep rank sweep: tamper_rank{N}_1ep_{config} dirs and ta-rk{N}-{short} logs
    for tag in sweep_tags:
        dirname = f"tamper_rank{rank}_1ep_{tag}"
        data = load_results(str(runs_dir / dirname / "tamper_results_*.json"))
        if data:
            peak = max(r["wmdp_bio_acc"] for r in data)
            if peak > best_peak:
                best_peak, best_data = peak, data
                best_name, best_is_1ep = f"1ep_rank{rank}_{tag}", True

    for stag, sshort in sweep_short_map.items():
        for logf in glob.glob(str(runs_dir / f"ta-rk{rank}-{sshort}-*.out")):
            data = load_results_from_log(logf)
            if data:
                peak = max(r["wmdp_bio_acc"] for r in data)
                if peak > best_peak:
                    best_peak, best_data = peak, data
                    best_name, best_is_1ep = f"1ep_rank{rank}_{stag}", True

    return best_data, best_name, best_peak, best_is_1ep


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = repo_root / "runs"
    out_dir = repo_root / "experiment_logs"

    ranks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    rank_data = {}
    for r in ranks:
        data, name, peak, is_1ep = find_best_config(runs_dir, r)
        if data:
            rank_data[r] = (data, name, is_1ep)
            tag = " [1ep]" if is_1ep else " [multi-ep]"
            print(f"r={r:>3d}: best={name:<35s} peak={peak*100:.1f}%{tag}")

    # Load all filtered model results, pick best
    filtered_dirs = sorted(glob.glob(str(runs_dir / "tamper_filtered_*")))
    best_filtered = None
    best_filtered_name = None
    best_peak = 0
    for d in filtered_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        data = load_results(str(d / "tamper_results_*.json"))
        if data:
            peak = max(r["wmdp_bio_acc"] for r in data)
            if peak > best_peak:
                best_peak = peak
                best_filtered = data
                best_filtered_name = d.name
    if best_filtered:
        print(f"filt : best={best_filtered_name:<35s} peak={best_peak*100:.1f}%")

    # Color map for ranks
    cmap = plt.colormaps["viridis"]  # type: ignore

    fig, (ax_wmdp, ax_mmlu) = plt.subplots(2, 1, figsize=(12, 11), sharex=True)

    max_step = 0
    for i, r in enumerate(ranks):
        if r not in rank_data:
            continue
        data, name, is_1ep = rank_data[r]
        steps = [d["step"] for d in data]
        wmdp_accs = [d["wmdp_bio_acc"] * 100 for d in data]
        mmlu_accs = [d.get("mmlu_acc", float("nan")) * 100 for d in data]
        max_step = max(max_step, max(steps))
        color = cmap(0.15 + 0.75 * i / (len(ranks) - 1))
        lw = 2.0 if r in (2, 4, 32, 512) else 1.2
        marker = "s" if is_1ep else "o"
        label = f"r={r} ({name})"
        if is_1ep:
            label += " [1ep]"
        ax_wmdp.plot(
            steps,
            wmdp_accs,
            color=color,
            linewidth=lw,
            alpha=0.85,
            marker=marker,
            markersize=4,
            label=label,
        )
        has_mmlu = any(d.get("mmlu_acc") for d in data)
        if has_mmlu:
            ax_mmlu.plot(
                steps,
                mmlu_accs,
                color=color,
                linewidth=lw,
                alpha=0.85,
                marker=marker,
                markersize=4,
                label=label,
            )

    if best_filtered:
        steps = [d["step"] for d in best_filtered]
        wmdp_accs = [d["wmdp_bio_acc"] * 100 for d in best_filtered]
        mmlu_accs = [d.get("mmlu_acc", float("nan")) * 100 for d in best_filtered]
        max_step = max(max_step, max(steps))
        ax_wmdp.plot(
            steps,
            wmdp_accs,
            color="#e41a1c",
            linewidth=2.5,
            linestyle="--",
            label="Filtered model",
            zorder=10,
        )
        has_mmlu = any(d.get("mmlu_acc") for d in best_filtered)
        if has_mmlu:
            ax_mmlu.plot(
                steps,
                mmlu_accs,
                color="#e41a1c",
                linewidth=2.5,
                linestyle="--",
                label="Filtered model",
                zorder=10,
            )

    ax_wmdp.axhline(
        y=42.97,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Baseline (42.97%)",
    )
    ax_wmdp.axhline(
        y=25, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random (25%)"
    )
    ax_mmlu.axhline(
        y=44.76,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Baseline MMLU (44.8%)",
    )
    ax_mmlu.axhline(
        y=25, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random (25%)"
    )

    ax_wmdp.set_ylabel("WMDP Bio Accuracy (%)", fontsize=12)
    ax_wmdp.set_title(
        "Tamper Attack Recovery by LoRA Rank (ret=0, 1 epoch)", fontsize=13
    )
    ax_wmdp.set_ylim(20, 50)
    ax_wmdp.grid(True, alpha=0.3)
    ax_wmdp.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    ax_mmlu.set_xlabel("Finetuning Steps", fontsize=12)
    ax_mmlu.set_ylabel("MMLU Accuracy (%)", fontsize=12)
    ax_mmlu.set_title("MMLU During Tamper Attack", fontsize=13)
    ax_mmlu.set_xlim(0, max(3600, max_step + 100))
    ax_mmlu.set_ylim(15, 50)
    ax_mmlu.grid(True, alpha=0.3)
    ax_mmlu.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "tamper_rank_sweep_wmdp.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Summary table
    print(
        f"\n{'Rank':>6}  {'Config':<35s}  {'Start':>7}  "
        f"{'Peak':>7}  {'End':>7}  {'Recovery':>9}  {'1ep?':>5}"
    )
    print("-" * 95)
    for r in ranks:
        if r not in rank_data:
            continue
        data, name, is_1ep = rank_data[r]
        start = data[0]["wmdp_bio_acc"] * 100
        peak = max(d["wmdp_bio_acc"] for d in data) * 100
        end = data[-1]["wmdp_bio_acc"] * 100
        ep_tag = "yes" if is_1ep else "no"
        print(
            f"{r:>6}  {name:<35s}  {start:>6.1f}%  "
            f"{peak:>6.1f}%  {end:>6.1f}%  "
            f"{peak-start:>+7.1f}pp  {ep_tag:>5}"
        )

    if best_filtered:
        start = best_filtered[0]["wmdp_bio_acc"] * 100
        peak = max(d["wmdp_bio_acc"] for d in best_filtered) * 100
        end = best_filtered[-1]["wmdp_bio_acc"] * 100
        print(
            f"{'filt':>6}  {best_filtered_name:<35s} "
            f" {start:>6.1f}%  {peak:>6.1f}%  "
            f"{end:>6.1f}%  {peak-start:>+7.1f}pp    n/a"
        )


if __name__ == "__main__":
    main()
