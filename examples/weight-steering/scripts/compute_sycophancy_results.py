#!/usr/bin/env python
"""Normalize the newest Inspect sycophancy log into a lilpipe artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from inspect_ai.log import list_eval_logs, read_eval_log

from inspect_tasks.sycophancy_metrics import aggregate_records


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: compute_sycophancy_results.py OUTPUT_DIR")
    output = Path(sys.argv[1])
    logs = sorted(
        list_eval_logs(str(output)), key=lambda item: item.mtime or 0, reverse=True
    )
    if not logs:
        raise SystemExit(f"no Inspect logs found in {output}")
    log = read_eval_log(logs[0].name)
    records: list[dict[str, object]] = []
    for sample in log.samples or ():
        score = (sample.scores or {}).get("factual_correctness")
        if score is None:
            metadata = dict(sample.metadata or {})
            status = "judge_error"
            correct = None
        else:
            metadata = dict(score.metadata or sample.metadata or {})
            status = str(metadata.get("response_status", "judge_error"))
            correct = score.value == "C" if status == "valid" else None
        records.append(
            {
                "question": metadata["question"],
                "source": metadata["source"],
                "cue": metadata["cue"],
                "status": status,
                "correct": correct,
            }
        )
    metrics = aggregate_records(records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps({"results": {"sycophancy": metrics}}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
