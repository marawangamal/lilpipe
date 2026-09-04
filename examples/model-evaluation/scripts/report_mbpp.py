"""Print MBPP accuracy and hardcode rate from an lm-eval output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidates = sorted(args.output.glob("**/results*.json"))
    if not candidates:
        raise FileNotFoundError(f"No lm-eval results found below {args.output}")
    payload = json.loads(candidates[-1].read_text())
    metrics = payload["results"]["mbpp_evalplus"]
    print(f"MBPP accuracy: {metrics['pass_at_1,none']:.3%}")
    print(f"MBPP hardcode rate: {metrics['hardcode,none']:.3%}")
