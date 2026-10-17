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
    assert '("z7b-base", "Base model", "Base model", True)' in source
    assert '("minicpm5-1b-base", "Base model", "Base model", True)' in source
    assert 'palette["Base model"] = "black"' in source
    assert 'line.get_color() == "black"' in source
    assert 'line.set_marker("")' in source
    assert 'label.get_text() == "Base model"' in source
    assert 'handle.set_marker("")' in source
    assert "failed_criterion" not in source
    assert "Line2D" not in source
    assert "plot_trajectories" not in source
    assert "mmlu_path" not in source
    assert "relearning-attack-capability-filtered" in source
    assert '"model": "z7b",' not in source
    assert '"model": "z7b--2pp"' in source
    assert '"rootdir": "artifacts/tamia/evals/thresholds/2pp"' in source
    assert '"output": "relearning-attack-z7b--2pp"' in source
    assert '"model": "z7b--10pp"' in source
    assert '"rootdir": "artifacts/tamia/evals/thresholds/10pp"' in source
    assert '"output": "relearning-attack-z7b--10pp"' in source
    assert '"z7b-wmdp-bio--15pp--unlearn-fft-rmu--relearn-fft"' in source
    assert (
        '"z7b-wmdp-bio--15pp--unlearn-fft-rmu-sam-rho1e-2--relearn-fft"'
        in source
    )
    assert '"model": "optml-z7b--15pp"' in source
    assert '"output": "relearning-attack-optml-z7b--15pp"' in source
    assert '"optml-z7b-wmdp-bio--15pp--unlearn-fft-gd--relearn-fft"' in source
    assert (
        '"optml-z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2--relearn-fft"' in source
    )
    assert '"optml-z7b-wmdp-bio--15pp--unlearn-fft-npo--relearn-fft"' in source
    assert (
        '"optml-z7b-wmdp-bio--15pp--unlearn-fft-npo-sam--relearn-fft"' in source
    )
    assert '"base_rootdir": "artifacts/tamia/evals"' in source
    assert '"steps": range(0, 17, 4)' in source
    assert 'steps = experiment.get("steps", range(0, 126, 25))' in source
    assert 'parser.add_argument("--output"' in source
    assert 'parser.add_argument("--out"' not in source
    assert "No measured trajectories found" in source
