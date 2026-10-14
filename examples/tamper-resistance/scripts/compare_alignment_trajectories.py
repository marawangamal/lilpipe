"""Create a Figure 3-style SFT versus Booster comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_alignment import extract_relearning_losses


def load_trajectory(path, method):
    """Load and validate one ordered held-out harmful-score trajectory."""
    rows = json.loads((Path(path) / "trajectory.json").read_text())
    if not rows or any(row["method"] != method for row in rows):
        raise ValueError(f"invalid {method} trajectory: {path}")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(set(steps)) or steps[0] != 0:
        raise ValueError(
            f"trajectory steps must be unique, sorted, and start at zero: {path}"
        )
    return rows


def rolling_mean(rows, window=20):
    """Smooth per-step training NLL without changing its step coordinates."""
    values = [row["harmful_nll"] for row in rows]
    smoothed = []
    for index, row in enumerate(rows):
        start = max(0, index - window + 1)
        smoothed.append(
            {
                "step": row["step"],
                "harmful_nll": sum(values[start : index + 1]) / (index - start + 1),
            }
        )
    return smoothed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("sft_trajectory", type=Path)
    parser.add_argument("booster_trajectory", type=Path)
    parser.add_argument("sft_relearned_adapter", type=Path)
    parser.add_argument("booster_relearned_adapter", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    trajectories = {
        "SFT": load_trajectory(args.sft_trajectory, "SFT"),
        "Booster": load_trajectory(args.booster_trajectory, "Booster"),
    }
    losses = {
        "SFT": extract_relearning_losses(args.sft_relearned_adapter),
        "Booster": extract_relearning_losses(args.booster_relearned_adapter),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    combined_rows = [row for rows in trajectories.values() for row in rows]
    (args.output / "heldout-harmful-trajectory.json").write_text(
        json.dumps(combined_rows, indent=2) + "\n"
    )
    with (args.output / "heldout-harmful-trajectory.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "step",
                "harmful_rate",
                "harmful_percent",
                "heldout_harmful_nll",
            ),
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method, rows in trajectories.items():
        axes[0].plot(
            [row["step"] for row in rows],
            [row["harmful_percent"] for row in rows],
            marker="o",
            label=method,
        )
    axes[0].set(
        xlabel="Harmful-only relearning step",
        ylabel="Held-out harmful responses (%)",
        ylim=(0, 100),
    )
    axes[0].legend()

    for method, rows in losses.items():
        smooth = rolling_mean(rows)
        axes[1].plot(
            [row["step"] for row in smooth],
            [row["harmful_nll"] for row in smooth],
            label=method,
        )
    axes[1].set(
        xlabel="Harmful-only relearning step",
        ylabel="Harmful training NLL (20-step mean)",
    )
    axes[1].legend()
    for method, rows in trajectories.items():
        axes[2].plot(
            [row["step"] for row in rows],
            [row["heldout_harmful_nll"] for row in rows],
            marker="o",
            label=method,
        )
    axes[2].set(
        xlabel="Harmful-only relearning step",
        ylabel="Held-out harmful NLL",
    )
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(args.output / "figure3-style-comparison.png", dpi=200)
    figure.savefig(args.output / "figure3-style-comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
