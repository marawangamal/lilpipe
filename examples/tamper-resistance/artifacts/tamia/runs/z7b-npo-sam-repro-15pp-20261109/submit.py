"""Submit the fresh reproduction after its allocated preflight passes."""

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path("../../src").resolve()))
from lilpipe import load, submit

run = Path("artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109")
preflight = json.loads((run / "preflight-job.json").read_text())["job_id"]
state = subprocess.check_output(
    ["sacct", "-j", preflight, "-X", "-n", "--format=State,ExitCode"], text=True
).split()
assert state == ["COMPLETED", "0:0"], state
smokes = json.loads((run / "smoke-verification.json").read_text())
assert len(smokes) == 2 and all(
    item["step"] == 1 and item["model_only"] for item in smokes
)
assert (run / "eval-cache-verification.json").exists()
receipt = run / "jobs.json"
assert not receipt.exists(), "Inspect existing job receipts instead of resubmitting"
for method in ["npo", "npo-sam-rho1e-2"]:
    for ending in ["", "--relearn-fft"]:
        name = f"z7b-wmdp-bio--15pp--unlearn-fft-{method}{ending}"
        assert not (Path("artifacts/tamia/models") / name).exists(), name
        assert not (Path("artifacts/tamia/evals") / name).exists(), name
plan = load("configs/experiments/z7b-wmdp-bio-npo-sam-repro--15pp.yml").plan()
(run / "dry-run.txt").write_text(
    plan.render(extra_sbatch_args=["--account=aip-grabus"]) + "\n"
)
ids = {}
commands = {}


def launch(command):
    stage = next(
        arg.split("=", 1)[1] for arg in command if arg.startswith("--job-name=")
    )
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    assert job.isdigit(), job
    ids[stage] = job
    commands[stage] = command
    receipt.write_text(json.dumps(ids, indent=2) + "\n")
    (run / "submitted-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return job


submit(plan, extra_sbatch_args=["--account=aip-grabus"], run=launch)
print(json.dumps(ids, indent=2))
