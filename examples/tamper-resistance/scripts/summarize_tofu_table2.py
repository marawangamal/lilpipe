#!/usr/bin/env python3
"""Combine basic and attack metrics into the GD/SSIUU Table 2 subset."""

import argparse
import json
from pathlib import Path
from statistics import mean

PAPER_RESULTS = {
    "GD": {
        "FS": 19.32,
        "RS": 89.36,
        "US": 92.88,
        "harmful_p0.1": 45.41,
        "harmful_p0.3": 51.73,
        "benign": 20.98,
    },
    "SSIUU": {
        "FS": 17.75,
        "RS": 92.74,
        "US": 91.67,
        "harmful_p0.1": 31.82,
        "harmful_p0.3": 37.53,
        "benign": 21.08,
    },
}


def attack_score(root: Path, method: str, attack: str, fraction=None):
    rows = []
    for path in (root / method / attack).glob("**/results.json"):
        row = json.loads(path.read_text())
        if fraction is None or row["fraction"] == fraction:
            rows.append(row)
    per_seed = []
    for seed in (1, 2, 3):
        candidates = [row["FS"] for row in rows if row["seed"] == seed]
        if not candidates:
            raise RuntimeError(f"missing {method} {attack} results for seed {seed}")
        per_seed.append(max(candidates))
    return mean(per_seed)


def markdown_table(reproduced):
    columns = (
        ("FS", "FS"),
        ("RS", "RS"),
        ("US", "US"),
        ("Harmful p=0.1", "harmful_p0.1"),
        ("Harmful p=0.3", "harmful_p0.3"),
        ("Benign", "benign"),
    )
    lines = [
        "| Method | Source | " + " | ".join(label for label, _ in columns) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    reproduced_by_method = {row["method"]: row for row in reproduced}
    for method in ("GD", "SSIUU"):
        for source, row in (
            ("Paper", PAPER_RESULTS[method]),
            ("Reproduction", reproduced_by_method[method]),
        ):
            values = " | ".join(f"{row[key]:.2f}" for _, key in columns)
            lines.append(f"| {method} | {source} | {values} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("basic_root", type=Path)
    parser.add_argument("attack_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    table = []
    for short_method in ("gd", "ssiuu"):
        method = f"llama32-tofu--unlearn-fft-{short_method}"
        basic = json.loads(
            (args.basic_root / method / "tofu-table2/results.json").read_text()
        )
        table.append(
            {
                "method": "GD" if short_method == "gd" else "SSIUU",
                "FS": basic["FS"],
                "RS": basic["RS"],
                "US": basic["US"],
                "harmful_p0.1": attack_score(args.attack_root, method, "harmful", 0.1),
                "harmful_p0.3": attack_score(args.attack_root, method, "harmful", 0.3),
                "benign": attack_score(args.attack_root, method, "benign"),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"paper": PAPER_RESULTS, "reproduction": table}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown_table(table))


if __name__ == "__main__":
    main()
