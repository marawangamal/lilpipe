from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "examples"
    / "tamper-resistance"
    / "scripts"
    / "analysis"
    / "plot_trajectory.py"
)


def test_trajectory_plot_uses_one_tidy_seaborn_plot():
    source = SCRIPT.read_text()

    assert source.count("sns.relplot(") == 1
    assert 'hue="Unlearning Method"' in source
    assert 'style="Tamper Resistance Method"' in source
    assert 'col="eval"' in source
    assert "failed_criterion" not in source
    assert "Line2D" not in source
    assert "plot_trajectories" not in source
    assert "mmlu_path" not in source
    assert "relearning-attack-capability-filtered" in source
