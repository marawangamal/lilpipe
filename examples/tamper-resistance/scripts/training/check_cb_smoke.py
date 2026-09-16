"""Fail unless a circuit-breaker smoke run demonstrates learned rerouting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def learned_lora_b(adapter_path: Path) -> bool:
    """Return whether at least one learned LoRA-B tensor is nonzero."""

    with safe_open(adapter_path, framework="pt", device="cpu") as weights:
        names = [name for name in weights.keys() if "lora_B" in name]
        if not names:
            raise ValueError(f"no LoRA-B tensors found in {adapter_path}")
        return any(bool(weights.get_tensor(name).count_nonzero()) for name in names)


def harmful_cosines(state_path: Path) -> list[float]:
    """Read finite harmful-cosine measurements from Trainer state."""

    state = json.loads(state_path.read_text())
    values = [
        float(row["harmful_cosine"])
        for row in state.get("log_history", [])
        if "harmful_cosine" in row
    ]
    if len(values) < 2:
        raise ValueError(f"fewer than two harmful_cosine measurements in {state_path}")
    return values


def validate_smoke_run(output_dir: Path, minimum_decline: float = 1e-3) -> float:
    """Validate adapter updates and return the harmful-cosine decline."""

    checkpoint = output_dir / "checkpoint-25"
    if not learned_lora_b(checkpoint / "adapter_model.safetensors"):
        raise RuntimeError("LoRA-B weights remained zero; the adapter did not update")
    cosines = harmful_cosines(checkpoint / "trainer_state.json")
    decline = cosines[0] - cosines[-1]
    if decline < minimum_decline:
        raise RuntimeError(
            "harmful cosine remained effectively unchanged: "
            f"initial={cosines[0]:.6f}, final={cosines[-1]:.6f}, "
            f"decline={decline:.6f}, required={minimum_decline:.6f}"
        )
    return decline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-decline", type=float, default=1e-3)
    args = parser.parse_args()
    decline = validate_smoke_run(args.output_dir, args.minimum_decline)
    print(f"CB smoke safeguard passed: harmful cosine declined by {decline:.6f}")


if __name__ == "__main__":
    main()
