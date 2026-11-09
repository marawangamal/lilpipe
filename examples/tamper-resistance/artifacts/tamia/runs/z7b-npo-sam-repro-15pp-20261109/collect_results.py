"""Verify and collect the fresh 15pp NPO/SAM reproduction."""

import csv
import json
import math
from pathlib import Path

run = Path("artifacts/tamia/runs/z7b-npo-sam-repro-15pp-20261109")
rows = []
checkpoints = []
for method, label in [("npo", "NPO"), ("npo-sam-rho1e-2", "NPO + SAM")]:
    defense = f"z7b-wmdp-bio--15pp--unlearn-fft-{method}"
    attack = defense + "--relearn-fft"
    for name, steps in [(defense, [125]), (attack, [4, 8, 12, 16])]:
        for step in steps:
            checkpoint = Path("artifacts/tamia/models") / name / f"checkpoint-{step}"
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            assert state["global_step"] == step, checkpoint
            weights = list(checkpoint.glob("*.safetensors"))
            weight_bytes = sum(path.stat().st_size for path in weights)
            assert weight_bytes > 14_000_000_000, checkpoint
            assert not (checkpoint / "optimizer.pt").exists(), checkpoint
            checkpoints.append(
                {
                    "path": str(checkpoint),
                    "global_step": step,
                    "weight_bytes": weight_bytes,
                }
            )
    for step in [0, 4, 8, 12, 16]:
        expected_model = (
            f"artifacts/tamia/models/{defense}"
            if step == 0
            else f"artifacts/tamia/models/{attack}/checkpoint-{step}"
        )
        directory = Path("artifacts/tamia/evals") / attack / f"checkpoint-{step}"
        files = sorted(directory.rglob("results*.json"))
        assert len(files) == 1, (directory, files)
        result = json.loads(files[0].read_text())
        assert result["config"].get("limit") is None, files[0]
        args = result["config"]["model_args"]
        if isinstance(args, str):
            args = dict(part.split("=", 1) for part in args.split(",") if "=" in part)
        assert Path(args["pretrained"]).resolve(strict=True) == Path(
            expected_model
        ).resolve(strict=True), (args, expected_model)
        scores = result["results"]
        wmdp = 100 * scores["wmdp_bio_robust"]["acc,none"]
        mmlu = 100 * scores["mmlu_no_bio"]["acc,none"]
        assert math.isfinite(wmdp) and math.isfinite(mmlu)
        rows.append(
            {
                "method": label,
                "step": step,
                "wmdp_bio_robust": wmdp,
                "mmlu_no_bio": mmlu,
                "source": str(files[0]),
            }
        )
(run / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
(run / "checkpoint-verification.json").write_text(
    json.dumps(checkpoints, indent=2) + "\n"
)
with (run / "trajectory.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
