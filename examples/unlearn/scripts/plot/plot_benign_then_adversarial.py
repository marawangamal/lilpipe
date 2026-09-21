"""Plot adversarial tamper recovery on benign-tampered checkpoints (Seq + CT).

For each model (seq_benign_final, ct_benign_final), picks the best tamper config
by peak WMDP. Two subplots: WMDP and MMLU.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUTPUT_PATH = REPO_ROOT / "experiment_logs" / "benign_then_adversarial_tamper.png"

MODEL_TAGS = ["seq_benign_final", "ct_benign_final"]
DISPLAY_NAMES = {
    "seq_benign_final": "Seq (after benign tamper)",
    "ct_benign_final": "CT (after benign tamper)",
}

_TAB10 = plt.cm.tab10(np.linspace(0, 1, 10))
MODEL_COLORS = {
    "seq_benign_final": _TAB10[0],
    "ct_benign_final": _TAB10[1],
}


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
        step_str = f.stem.split("_")[1]
        try:
            step = int(step_str)
            data = json.loads(f.read_text())
            results[step] = data
        except (ValueError, json.JSONDecodeError, IndexError):
            continue
    return results


def get_best_config_per_group() -> dict[str, tuple[list[int], list[float], list[float], str]]:
    groups: dict[str, list[tuple[str, dict[int, dict]]]] = {}

    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("tamper_"):
            continue
        info = parse_tamper_dirname(d.name)
        if info is None or info["model_tag"] not in MODEL_TAGS:
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
            wmdp_accs = [best_data[s].get("wmdp_bio_acc", 0) * 100 for s in steps_sorted]
            mmlu_accs = [best_data[s].get("mmlu_acc", 0) * 100 for s in steps_sorted]
            best_per_group[model_tag] = (steps_sorted, wmdp_accs, mmlu_accs, best_label)

    return best_per_group


def main():
    best_per_group = get_best_config_per_group()

    if not best_per_group:
        print("No benign→adversarial tamper data found.")
        return

    fig, (ax_wmdp, ax_mmlu) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for tag in MODEL_TAGS:
        if tag not in best_per_group:
            continue
        steps, wmdp_accs, mmlu_accs, config_label = best_per_group[tag]
        name = DISPLAY_NAMES[tag]
        color = MODEL_COLORS[tag]

        ax_wmdp.plot(
            steps, wmdp_accs,
            color=color, linewidth=2, alpha=0.85,
            marker="o", markersize=4,
            label=f"{name} ({config_label})",
        )
        ax_mmlu.plot(
            steps, mmlu_accs,
            color=color, linewidth=2, alpha=0.85,
            marker="o", markersize=4,
            label=f"{name} ({config_label})",
        )

    # Reference lines
    ax_wmdp.axhline(y=42.97, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
                     label="Baseline WMDP (42.97%)")
    ax_wmdp.axhline(y=25, color="green", linestyle=":", linewidth=1, alpha=0.5,
                     label="Random (25%)")
    ax_mmlu.axhline(y=45.10, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
                     label="Baseline MMLU (45.10%)")
    ax_mmlu.axhline(y=25, color="green", linestyle=":", linewidth=1, alpha=0.5,
                     label="Random (25%)")

    ax_wmdp.set_ylabel("WMDP Bio Robust Accuracy (%)", fontsize=12)
    ax_wmdp.set_title(
        "Adversarial Tampering After Benign Fine-tuning\n"
        "(Unlearn → 10k benign steps → bio_remove attack)",
        fontsize=13,
    )
    ax_wmdp.legend(loc="upper left", fontsize=9)
    ax_wmdp.grid(True, alpha=0.3)
    ax_wmdp.set_ylim(20, 50)

    ax_mmlu.set_xlabel("Adversarial Tampering Step", fontsize=12)
    ax_mmlu.set_ylabel("MMLU Accuracy (%)", fontsize=12)
    ax_mmlu.set_title("MMLU During Adversarial Tampering", fontsize=13)
    ax_mmlu.legend(loc="upper left", fontsize=9)
    ax_mmlu.grid(True, alpha=0.3)
    ax_mmlu.set_ylim(20, 50)
    ax_mmlu.set_xlim(-100, 10500)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_PATH}")

    for tag in MODEL_TAGS:
        if tag not in best_per_group:
            continue
        steps, wmdp_accs, mmlu_accs, config_label = best_per_group[tag]
        print(f"\n{DISPLAY_NAMES[tag]} ({config_label}):")
        print(f"  WMDP: {wmdp_accs[0]:.1f}% → {max(wmdp_accs):.1f}% (peak) → {wmdp_accs[-1]:.1f}% (final)")
        print(f"  MMLU: {mmlu_accs[0]:.1f}% → {mmlu_accs[-1]:.1f}% (final)")


if __name__ == "__main__":
    main()
