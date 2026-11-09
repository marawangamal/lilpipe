"""Validate generation artifacts without imposing an accuracy or formatting gate."""

import json
import sys
from pathlib import Path


def validate_smoke(directory: Path) -> int:
    files = sorted(directory.rglob("results*.json"))
    if not files:
        raise ValueError("Smoke evaluation did not save metrics")
    result = files[-1]
    data = json.loads(result.read_text())
    timestamp = result.stem.removeprefix("results_")
    samples = sorted(result.parent.glob(f"samples_*_{timestamp}.jsonl"))
    expected = sum(row["effective"] for row in data["n-samples"].values())
    count = 0
    for sample in samples:
        for line in sample.open():
            response = json.loads(line)["resps"]
            while isinstance(response, list) and len(response) == 1:
                response = response[0]
            if not isinstance(response, str):
                raise ValueError("Smoke response is not a generated text record")
            count += 1
    if not count or count != expected:
        raise ValueError(
            f"Incomplete smoke artifacts: {count} responses, expected {expected}"
        )
    return count


if __name__ == "__main__":
    count = validate_smoke(Path(sys.argv[1]))
    print(
        f"Generation saved {count} responses; invalid answers remain incorrect.",
        flush=True,
    )
