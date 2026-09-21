"""Plot WMDP and MMLU adversarial tamper curves for fp32 unlearned models (Seq + CT)."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUTPUT_DIR = REPO_ROOT / "experiment_logs"

FP32_TAGS = {
    "seq_sft_ret0_rm5_lr2e-4_nn2_fp32": "Seq ret0 rm5 (fp32)",
    "ct_sft_ret0_rm2000_lr5e-4_muon_fp32": "CT Muon ret0 rm2000 (fp32)",
}

_TAB10 = plt.colormaps["tab10"](np.linspace(0, 1, 10))
MODEL_COLORS = {tag: _TAB10[i] for i, tag in enumerate(FP32_TAGS)}


def parse_tamper_dirname(dirname: str, data_type: str = "bio_remove") -> dict | None:
    pattern = re.compile(
        r"tamper_(.+?)_" + re.escape(data_type) + r"_lr([\d.e-]+)_s(\d+)_(\w+)_(\w+)"
    )
    m = pattern.match(dirname)
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


def get_best_config_per_group(
    data_type: str = "bio_remove",
    allowed_tags: dict | None = None,
) -> dict[str, tuple[list[int], list[float], list[float], str]]:
    groups: dict[str, list[tuple[str, dict[int, dict]]]] = {}

    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("tamper_"):
            continue
        info = parse_tamper_dirname(d.name, data_type)
        if info is None:
            continue
        if allowed_tags and info["model_tag"] not in allowed_tags:
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
            wmdp_accs = [
                best_data[s].get("wmdp_bio_acc", 0) * 100 for s in steps_sorted
            ]
            mmlu_accs = [best_data[s].get("mmlu_acc", 0) * 100 for s in steps_sorted]
            best_per_group[model_tag] = (steps_sorted, wmdp_accs, mmlu_accs, best_label)

    return best_per_group


def plot_two_panel(
    best_per_group: dict,
    display_names: dict,
    colors: dict,
    title_prefix: str,
    output_path: Path,
):
    if not best_per_group:
        print(f"No data for {output_path.name}")
        return

    fig, (ax_wmdp, ax_mmlu) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for tag in display_names:
        if tag not in best_per_group:
            continue
        steps, wmdp_accs, mmlu_accs, config_label = best_per_group[tag]
        name = display_names[tag]
        color = colors[tag]

        if len(steps) > 50:
            indices = [0]
            for i in range(1, len(steps)):
                if steps[i] - steps[indices[-1]] >= 450:
                    indices.append(i)
            if indices[-1] != len(steps) - 1:
                indices.append(len(steps) - 1)
            steps = [steps[i] for i in indices]
            wmdp_accs = [wmdp_accs[i] for i in indices]
            mmlu_accs = [mmlu_accs[i] for i in indices]

        ax_wmdp.plot(
            steps,
            wmdp_accs,
            color=color,
            linewidth=2,
            alpha=0.85,
            marker="o",
            markersize=4,
            label=f"{name} ({config_label})",
        )
        ax_mmlu.plot(
            steps,
            mmlu_accs,
            color=color,
            linewidth=2,
            alpha=0.85,
            marker="o",
            markersize=4,
            label=f"{name} ({config_label})",
        )

    for ax, metric, baseline, blabel in [
        (ax_wmdp, "WMDP Bio Robust Accuracy (%)", 42.97, "Baseline WMDP (42.97%)"),
        (ax_mmlu, "MMLU Accuracy (%)", 45.10, "Baseline MMLU (45.10%)"),
    ]:
        ax.axhline(
            y=baseline,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=blabel,
        )
        ax.axhline(
            y=25,
            color="green",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label="Random (25%)",
        )
        ax.set_ylabel(metric, fontsize=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(15, 50)

    ax_wmdp.set_title(f"{title_prefix} (fp32 Models)", fontsize=13)
    ax_mmlu.set_xlabel("Tampering Step", fontsize=12)
    ax_mmlu.set_xlim(-100, 11000)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")

    for tag in display_names:
        if tag not in best_per_group:
            continue
        steps, wmdp_accs, mmlu_accs, config_label = best_per_group[tag]
        print(f"  {display_names[tag]} ({config_label}):")
        print(
            f"    WMDP: {wmdp_accs[0]:.1f}% → {max(wmdp_accs):.1f}% (peak) → {wmdp_accs[-1]:.1f}%"
        )
        print(f"    MMLU: {mmlu_accs[0]:.1f}% → {mmlu_accs[-1]:.1f}%")


def main():
    # Adversarial tamper (bio_remove)
    adversarial = get_best_config_per_group("bio_remove", FP32_TAGS)
    plot_two_panel(
        adversarial,
        FP32_TAGS,
        MODEL_COLORS,
        "Adversarial Tampering (bio_remove)",
        OUTPUT_DIR / "fp32_tamper_adversarial.png",
    )

    # Benign tamper
    benign = get_best_config_per_group("benign", FP32_TAGS)
    plot_two_panel(
        benign,
        FP32_TAGS,
        MODEL_COLORS,
        "Benign Tampering",
        OUTPUT_DIR / "fp32_tamper_benign.png",
    )

    # Benign→adversarial tamper
    benign_adv_tags = {
        "seq_sft_ret0_rm5_lr2e-4_nn2_fp32_benign_final": "Seq (after benign, fp32)",
        "ct_sft_ret0_rm2000_lr5e-4_muon_fp32_benign_final": "CT Muon (after benign, fp32)",
    }
    benign_adv_colors = {tag: _TAB10[i] for i, tag in enumerate(benign_adv_tags)}
    benign_adv = get_best_config_per_group("bio_remove", benign_adv_tags)
    plot_two_panel(
        benign_adv,
        benign_adv_tags,
        benign_adv_colors,
        "Adversarial Tampering After Benign Fine-tuning",
        OUTPUT_DIR / "fp32_tamper_benign_then_adversarial.png",
    )


if __name__ == "__main__":
    main()
