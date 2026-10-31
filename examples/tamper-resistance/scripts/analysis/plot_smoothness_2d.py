"""Plot 2D loss contours and centered checkpoint trajectories from gen_smoothness.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_smoothness import merge_results, select_direction_std


def draw_trajectory(axis, points, labels, *, annotate=False):
    """Connect checkpoints in their saved order without changing their coordinates."""
    axis.plot(*points.T, color="crimson", linewidth=1.3, zorder=4)
    axis.scatter(*points.T, color="crimson", edgecolors="white", s=35, zorder=5)
    axis.scatter(*points[0], marker="*", color="black", s=110, zorder=6, label="Step 0")
    axis.scatter(
        *points[-1],
        marker="D",
        color="crimson",
        edgecolors="white",
        s=55,
        zorder=7,
        label=f"Step {labels[-1]}",
    )
    if annotate:
        # Later checkpoints can cluster tightly; label landmarks without crowding.
        for index in sorted({0, min(1, len(points) - 1), len(points) - 1}):
            axis.annotate(
                labels[index],
                points[index],
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )


def plot_surfaces(results, output, *, columns=4, levels=12):
    """Draw one contour panel per model/scale, with a zoom for tiny trajectories."""
    if columns < 1 or levels < 2:
        raise ValueError("columns must be positive and levels must be at least two")
    models = results["models"]
    if not models:
        raise ValueError("No surfaces to plot")
    coordinates = np.asarray(results["coordinates"], dtype=float)
    x, y = np.meshgrid(coordinates, coordinates)
    columns = min(columns, len(models))
    rows = (len(models) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.3 * columns, 4.7 * rows),
        squeeze=False,
        layout="constrained",
    )

    for axis, surface in zip(axes.flat, models):
        trajectory = surface.get("trajectory")
        if not trajectory:
            raise ValueError(
                "Each surface must include a centered, scale-specific trajectory"
            )
        loss = np.asarray(surface["loss"], dtype=float)
        points = np.asarray([point["coordinates"] for point in trajectory], dtype=float)
        if loss.shape != x.shape or points.shape != (len(trajectory), 2):
            raise ValueError("Loss grid or trajectory coordinates have the wrong shape")
        if not np.isfinite(loss).all() or not np.isfinite(points).all():
            raise ValueError("Losses and trajectory coordinates must be finite")
        labels = [
            (
                Path(point["path"]).name.removeprefix("checkpoint-")
                if Path(point["path"]).name.startswith("checkpoint-")
                else str(index)
            )
            for index, point in enumerate(trajectory)
        ]
        low, high = float(loss.min()), float(loss.max())
        if np.isclose(low, high):
            padding = max(abs(low), 1.0) * 0.01
            low, high = low - padding, high + padding
        contour_levels = np.linspace(low, high, levels)
        filled = axis.contourf(x, y, loss, levels=contour_levels, cmap="viridis")
        contours = axis.contour(
            x,
            y,
            loss,
            levels=contour_levels,
            colors="black",
            linewidths=0.45,
            alpha=0.55,
        )
        axis.clabel(contours, inline=True, fontsize=7, fmt="%.1f")
        figure.colorbar(
            filled, ax=axis, label="Forget loss", shrink=0.85, format="%.1f"
        )
        draw_trajectory(axis, points, labels)
        # Include points outside the sampled grid, leaving the unsampled area blank.
        axis.update_datalim(points)
        axis.autoscale_view()
        axis.set(
            title=f"{surface['label']} · σ={surface['direction_std']:.0e}",
            xlabel="x",
            ylabel="y",
            aspect="equal",
        )
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(False)

        if np.ptp(points, axis=0).max() < 0.01 * np.ptp(coordinates):
            zoom = axis.inset_axes([0.09, 0.58, 0.39, 0.32])
            draw_trajectory(zoom, points, labels, annotate=True)
            zoom.set_title("Trajectory (zoom)", fontsize=8)
            zoom.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
            zoom.tick_params(labelsize=6)
            for offset in (zoom.xaxis.get_offset_text(), zoom.yaxis.get_offset_text()):
                offset.set_fontsize(6)
            zoom.margins(0.3)
            zoom.grid(alpha=0.25)

    for axis in list(axes.flat)[len(models) :]:
        axis.set_visible(False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", dpi=200)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", metavar="JSON")
    parser.add_argument("--output", default="artifacts/figures/smoothness-2d.pdf")
    parser.add_argument("--direction-std", type=float, help="Plot only this scale")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--levels", type=int, default=12)
    args = parser.parse_args()

    results = merge_results(args.inputs)
    if args.direction_std is not None:
        results = select_direction_std(results, args.direction_std)
    sns.set_theme(style="white")
    figure = plot_surfaces(
        results, args.output, columns=args.columns, levels=args.levels
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
