"""Plot WMDP and MMLU learning curves for benign tamper runs."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/runs")
OUTPUT_DIR = Path("/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/experiment_logs")

TAMPER_DATA = "benign"
DIRNAME_PATTERN = re.compile(
    r"tamper_(.+?)_" + re.escape(TAMPER_DATA) + r"_lr([\d.e-]+)_s(\d+)_(\w+)_(\w+)"
)

E2E_FILTER_HASH = "b28797cd9b615104ba9d24e6900336253323e7cf"

_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20))
DISPLAY_NAMES = {
    "seq_sft_ret0_rm5_lr2e-4_nn2": "Seq ret0 rm5 lr2e-4",
    "ct_sft_muon_ret0_rm2000": "CT Muon ret0 rm2000",
    E2E_FILTER_HASH: "E2E Filter",
}
MODEL_COLORS = {tag: _TAB20[i % len(_TAB20)] for i, tag in enumerate(DISPLAY_NAMES)}


def parse_tamper_dirname(dirname: str) -> dict | None:
    m = DIRNAME_PATTERN.match(dirname)
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


def get_best_config_per_group() -> (
    dict[str, tuple[list[int], list[float], list[float], str]]
):
    """For each model group, find the tamper config with best peak WMDP.
    Returns {model_tag: (steps, wmdp_accs, mmlu_accs, config_label)}."""
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
            peak = max(
                (r.get("wmdp_bio_acc", 0) for r in results.values()), default=0
            )
            if peak > best_peak:
                best_peak = peak
                best_data = results
                best_label = label

        if best_data:
            steps_sorted = sorted(best_data.keys())
            wmdp_accs = [
                best_data[s].get("wmdp_bio_acc", 0) * 100 for s in steps_sorted
            ]
            mmlu_accs = [
                best_data[s].get("mmlu_acc", 0) * 100 for s in steps_sorted
            ]
            best_per_group[model_tag] = (steps_sorted, wmdp_accs, mmlu_accs, best_label)

    return best_per_group


def plot_metric(
    best_per_group: dict,
    metric_idx: int,
    ylabel: str,
    title: str,
    baseline_val: float,
    baseline_label: str,
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(14, 7))

    sorted_tags = sorted(
        best_per_group.keys(),
        key=lambda t: best_per_group[t][metric_idx][0]
        if best_per_group[t][metric_idx]
        else 0,
    )

    for tag in sorted_tags:
        if tag not in DISPLAY_NAMES:
            continue
        steps = best_per_group[tag][0]
        accs = best_per_group[tag][metric_idx]
        config_label = best_per_group[tag][3]
        name = DISPLAY_NAMES[tag]

        if len(steps) > 50:
            indices = [0]
            for i in range(1, len(steps)):
                if steps[i] - steps[indices[-1]] >= 450:
                    indices.append(i)
            if indices[-1] != len(steps) - 1:
                indices.append(len(steps) - 1)
            steps = [steps[i] for i in indices]
            accs = [accs[i] for i in indices]

        is_filter = tag == E2E_FILTER_HASH
        ax.plot(
            steps,
            accs,
            color=MODEL_COLORS[tag],
            linewidth=2.5 if is_filter else 2,
            alpha=0.9 if is_filter else 0.85,
            marker="D" if is_filter else "o",
            markersize=4,
            linestyle="--" if is_filter else "-",
            label=f"{name} ({config_label})",
        )

    ax.axhline(
        y=baseline_val,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=baseline_label,
    )
    ax.axhline(
        y=25, color="green", linestyle=":", linewidth=1, alpha=0.5, label="Random (25%)"
    )

    ax.set_xlabel("Tampering Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(15, 50)
    ax.set_xlim(-100, 11000)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main():
    best_per_group = get_best_config_per_group()

    if not best_per_group:
        print("No benign tamper results found yet.")
        return

    plot_metric(
        best_per_group,
        metric_idx=1,
        ylabel="WMDP Bio Robust Accuracy (%)",
        title="WMDP Under Benign Tampering: Best Config per Model",
        baseline_val=42.97,
        baseline_label="Baseline WMDP (42.97%)",
        output_path=OUTPUT_DIR / "benign_tamper_wmdp_comparison.png",
    )

    plot_metric(
        best_per_group,
        metric_idx=2,
        ylabel="MMLU Accuracy (%)",
        title="MMLU Under Benign Tampering: Best Config per Model",
        baseline_val=46.0,
        baseline_label="Baseline MMLU (46.0%)",
        output_path=OUTPUT_DIR / "benign_tamper_mmlu_comparison.png",
    )


if __name__ == "__main__":
    main()
