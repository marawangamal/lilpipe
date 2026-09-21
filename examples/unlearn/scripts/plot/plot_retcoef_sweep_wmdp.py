#!/usr/bin/env python3
"""Plot WMDP recovery curves for retain coefficient sweep tamper attacks.

For each retain coefficient, selects the best-performing tamper config
(highest peak WMDP) across all sweep configs. Uses 1-epoch data from
ta-{ret_short}-{sweep_short}-*.out log files and tamper_r32_ret*_1ep_* JSON.
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
    """Parse eval results from a SLURM .out log file."""
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


RETAIN_COEFS = [
    {"tag": "ret1e-9", "short": "r1e-9", "label": "ret=1e-9"},
    {"tag": "ret1e-7", "short": "r1e-7", "label": "ret=1e-7"},
    {"tag": "ret1e-5", "short": "r1e-5", "label": "ret=1e-5"},
    {"tag": "ret1e-4", "short": "r1e-4", "label": "ret=1e-4"},
    {"tag": "ret1e-3", "short": "r1e-3", "label": "ret=1e-3"},
    {"tag": "ret1e-2", "short": "r1e-2", "label": "ret=1e-2"},
    {"tag": "ret0.02", "short": "r0.02", "label": "ret=0.02"},
    {"tag": "ret0.03", "short": "r0.03", "label": "ret=0.03"},
    {"tag": "ret0.05", "short": "r0.05", "label": "ret=0.05"},
    {"tag": "ret0.07", "short": "r0.07", "label": "ret=0.07"},
    {"tag": "ret0.1", "short": "r0.1", "label": "ret=0.1"},
    {"tag": "ret0.2", "short": "r0.2", "label": "ret=0.2"},
    {"tag": "ret0.4", "short": "r0.4", "label": "ret=0.4"},
    {"tag": "ret0.8", "short": "r0.8", "label": "ret=0.8"},
    {"tag": "ret1", "short": "r1", "label": "ret=1"},
]

SWEEP_CONFIGS = {
    "cosine_wr01_lr2e-5": "lr5",
    "cosine_wr01_lr2e-4": "std",
    "cosine_wr01_lr1e-3": "lr3",
    "cosine_ws30_lr2e-4": "ws30",
    "cosine_ws100_lr2e-4": "ws100",
    "constant_wr01_lr2e-4": "con",
}


def find_best_for_retcoef(runs_dir, ret):
    """Find best sweep config for a given retain coefficient."""
    best_data = None
    best_peak = -1
    best_cfg = None

    for stag, sshort in SWEEP_CONFIGS.items():
        dirname = f"tamper_r32_{ret['tag']}_1ep_{stag}"
        data = load_results(str(runs_dir / dirname / "tamper_results_*.json"))
        if not data:
            for logf in glob.glob(str(runs_dir / f"ta-{ret['short']}-{sshort}-*.out")):
                data = load_results_from_log(logf)
                if data:
                    break
        if data:
            peak = max(r["wmdp_bio_acc"] for r in data)
            if peak > best_peak:
                best_peak = peak
                best_data = data
                best_cfg = sshort

    return best_data, best_cfg, best_peak


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    runs_dir = repo_root / "runs"
    out_dir = repo_root / "experiment_logs"

    results = {}
    for ret in RETAIN_COEFS:
        data, cfg, peak = find_best_for_retcoef(runs_dir, ret)
        if data:
            results[ret["tag"]] = (data, cfg, peak, ret["label"])
            print(f"{ret['label']:>12s}: best={cfg:<6s} peak={peak*100:.1f}%")

    cmap = plt.colormaps["coolwarm"]  # type: ignore
    fig, ax = plt.subplots(figsize=(12, 7))

    max_step = 0
    for i, ret in enumerate(RETAIN_COEFS):
        if ret["tag"] not in results:
            continue
        data, cfg, peak, label = results[ret["tag"]]
        steps = [d["step"] for d in data]
        accs = [d["wmdp_bio_acc"] * 100 for d in data]
        max_step = max(max_step, max(steps))
        color = cmap(i / (len(RETAIN_COEFS) - 1))
        lw = 2.0 if ret["tag"] in ("ret1e-5", "ret1e-4", "ret0.02", "ret1") else 1.2
        ax.plot(
            steps,
            accs,
            color=color,
            linewidth=lw,
            alpha=0.85,
            marker="o",
            markersize=3,
            label=f"{label} ({cfg})",
        )

    # Filtered model tamper results
    filtered_dirs = sorted(glob.glob(str(runs_dir / "tamper_filtered_*")))
    best_filtered = None
    best_filtered_name = None
    best_filt_peak = 0
    for d in filtered_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        fdata = load_results(str(d / "tamper_results_*.json"))
        if fdata:
            fpeak = max(r["wmdp_bio_acc"] for r in fdata)
            if fpeak > best_filt_peak:
                best_filt_peak = fpeak
                best_filtered = fdata
                best_filtered_name = d.name
    if best_filtered:
        steps = [d["step"] for d in best_filtered]
        accs = [d["wmdp_bio_acc"] * 100 for d in best_filtered]
        max_step = max(max_step, max(steps))
        ax.plot(
            steps,
            accs,
            color="#e41a1c",
            linewidth=2.5,
            linestyle="--",
            label="Filtered model",
            zorder=10,
        )
        print(
            f"{'filtered':>12s}: best={best_filtered_name:<35s} "
            f"peak={best_filt_peak*100:.1f}%"
        )

    ax.axhline(
        y=42.97,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Baseline (42.97%)",
    )
    ax.axhline(
        y=25, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random (25%)"
    )

    ax.set_xlabel("Finetuning Steps", fontsize=12)
    ax.set_ylabel("WMDP Bio Accuracy (%)", fontsize=12)
    ax.set_title(
        "Tamper Attack Recovery by Retain Coefficient (r=32, 1 epoch)\n"
        "(best sweep config per ret coef)",
        fontsize=13,
    )
    ax.set_xlim(0, max(3600, max_step + 100))
    ax.set_ylim(20, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "tamper_retcoef_sweep_wmdp.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Summary table
    print(
        f"\n{'Ret Coef':>12s}  {'Config':<8s}  {'Start':>7}  {'Peak':>7}  "
        f"{'End':>7}  {'Recovery':>9}"
    )
    print("-" * 70)
    for ret in RETAIN_COEFS:
        if ret["tag"] not in results:
            continue
        data, cfg, peak, label = results[ret["tag"]]
        start = data[0]["wmdp_bio_acc"] * 100
        peak_pct = peak * 100
        end = data[-1]["wmdp_bio_acc"] * 100
        print(
            f"{label:>12s}  {cfg:<8s}  {start:>6.1f}%  {peak_pct:>6.1f}%  "
            f"{end:>6.1f}%  {peak_pct-start:>+7.1f}pp"
        )


if __name__ == "__main__":
    main()
