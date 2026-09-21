"""Scan tamper run eval_results directories and update the experiment log."""

import json
import re
import sys
from pathlib import Path

RUNS_DIR = Path("/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/runs")
LOG_PATH = Path(
    "/lus/lfs1aip2/projects/public/a6a/lucia/home/unlearn/experiment_logs/unrestrained_SFT.md"
)

# Standard step intervals for tables (every 500 steps)
STANDARD_STEPS = list(range(0, 10501, 500))


def parse_tamper_dirname(dirname: str) -> dict | None:
    """Extract metadata from a tamper directory name."""
    # e.g. tamper_lens_sft_ret0_rm100_lr1e-4_bio_remove_lr2e-5_s10000_linear_fp16
    # or   tamper_seq_sft_ret0_rm5_lr2e-4_nn2_bio_remove_lr2e-5_s10000_cosine_fp16
    # or   tamper_ct_sft_muon_ret0_rm2000_bio_remove_lr1e-5_s10000_linear_fp16
    m = re.match(r"tamper_(.+?)_bio_remove_lr([\d.e-]+)_s(\d+)_(\w+)_(\w+)", dirname)
    if not m:
        return None
    return {
        "model_tag": m.group(1),
        "tamper_lr": m.group(2),
        "max_steps": int(m.group(3)),
        "schedule": m.group(4),
        "dtype": m.group(5),
    }


def tamper_config_label(info: dict) -> str:
    """Create a short config label like 'linear/fp16/lr2e-5'."""
    return f"{info['schedule']}/{info['dtype']}/lr{info['tamper_lr']}"


def read_eval_results(eval_dir: Path) -> dict[int, dict]:
    """Read all step_*.json files from eval_results directory."""
    results = {}
    if not eval_dir.exists():
        return results
    for f in eval_dir.glob("step_*.json"):
        try:
            step = int(f.stem.split("_")[1])
            data = json.loads(f.read_text())
            results[step] = data
        except (ValueError, json.JSONDecodeError, IndexError):
            continue
    return results


def collect_tamper_runs() -> dict[str, list[dict]]:
    """Collect all tamper runs grouped by model_tag."""
    groups: dict[str, list[dict]] = {}
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("tamper_"):
            continue
        info = parse_tamper_dirname(d.name)
        if info is None:
            continue
        eval_dir = d / "eval_results"
        results = read_eval_results(eval_dir)
        if not results:
            continue
        info["results"] = results
        info["dirname"] = d.name
        groups.setdefault(info["model_tag"], []).append(info)
    return groups


def format_table(runs: list[dict]) -> str:
    """Format a markdown table for a group of tamper runs."""
    # Determine which steps have data across any run
    all_steps = set()
    for run in runs:
        all_steps.update(run["results"].keys())

    # Use standard steps that have data
    steps = sorted(s for s in STANDARD_STEPS if s in all_steps)
    if not steps:
        # Fall back to whatever steps exist, sampled at ~500 intervals
        raw_steps = sorted(all_steps)
        if not raw_steps:
            return ""
        steps = [raw_steps[0]]
        for s in raw_steps[1:]:
            if s - steps[-1] >= 450:
                steps.append(s)

    step_headers = " | ".join(f"Step {s}" for s in steps)
    header = f"| Config | Metric | {step_headers} |"
    separator = (
        "|--------|--------|"
        + "|".join("-" * (len(f"Step {s}") + 2) for s in steps)
        + "|"
    )

    rows = []
    for run in sorted(runs, key=lambda r: tamper_config_label(r)):
        label = tamper_config_label(run)
        wmdp_vals = []
        mmlu_vals = []
        for s in steps:
            r = run["results"].get(s)
            if r and "wmdp_bio_acc" in r:
                wmdp_vals.append(f"{r['wmdp_bio_acc'] * 100:.2f}")
            else:
                wmdp_vals.append("")
            if r and "mmlu_acc" in r:
                mmlu_vals.append(f"{r['mmlu_acc'] * 100:.2f}")
            else:
                mmlu_vals.append("")

        wmdp_row = f"| {label} | WMDP | " + " | ".join(wmdp_vals) + " |"
        mmlu_row = "| | MMLU | " + " | ".join(mmlu_vals) + " |"
        rows.append(wmdp_row)
        rows.append(mmlu_row)

    return "\n".join([header, separator] + rows)


# Model tag -> section header mapping
SECTION_HEADERS = {
    "lens_sft_ret0_rm100_lr1e-4": "### Lens SFT ret0 rm100 lr1e-4 Tamper",
    "seq_sft_ret0_rm5_lr2e-4_nn2": "### Sequential SFT ret0 rm5 nn2 Tamper",
    "lens_sft_ret5_rm5_lr1e-3": "### Lens SFT ret5 rm5 lr1e-3 Tamper",
    "lens_sft_ret5_rm5_lr1e-4": "### Lens SFT ret5 rm5 lr1e-4 Tamper",
    "lens_sft_ret5_rm5_lr5e-5": "### Lens SFT ret5 rm5 lr5e-5 Tamper",
    "lens_sft_ret5_rm5_lr2e-4": "### Lens SFT ret5 rm5 lr2e-4 Tamper",
    "lens_sft_ret0": "### Lens SFT ret0 Extended Tamper",
    "ct_sft_muon_ret0_rm2000": "### CT SFT Muon ret0 rm2000 Extended Tamper",
}


def update_log(groups: dict[str, list[dict]]) -> None:
    """Update the experiment log with new tamper results."""
    content = LOG_PATH.read_text()

    new_sections = []
    for model_tag, runs in sorted(groups.items()):
        header = SECTION_HEADERS.get(model_tag)
        if header is None:
            header = f"### {model_tag} Tamper"

        table = format_table(runs)
        if not table:
            continue

        # Check if section already exists
        if header in content:
            # Replace existing section (header through next ### or end of file)
            pattern = re.escape(header) + r"\n\n.*?(?=\n### |\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                tamper_info = _build_tamper_info(runs)
                replacement = f"{header}\n\n{tamper_info}\n\n{table}\n"
                content = (
                    content[: match.start()] + replacement + content[match.end() :]
                )
        else:
            tamper_info = _build_tamper_info(runs)
            new_sections.append(f"\n{header}\n\n{tamper_info}\n\n{table}\n")

    if new_sections:
        content = content.rstrip() + "\n" + "\n".join(new_sections) + "\n"

    LOG_PATH.write_text(content)
    print(f"Updated {LOG_PATH}")


def _build_tamper_info(runs: list[dict]) -> str:
    """Build tamper info line."""
    max_steps = runs[0]["max_steps"]
    n_configs = len(runs)
    return (
        f"Tamper: AdamW, {max_steps} steps, {n_configs} configs, evaluated every 500."
    )


def print_summary(groups: dict[str, list[dict]]) -> None:
    """Print a summary of available results."""
    for model_tag, runs in sorted(groups.items()):
        max_step = max(max(r["results"].keys()) for r in runs)
        n_results = sum(len(r["results"]) for r in runs)
        print(
            f"  {model_tag}: {len(runs)} configs, "
            f"{n_results} eval results, max step {max_step}"
        )


def main():
    dry_run = "--dry-run" in sys.argv

    groups = collect_tamper_runs()
    if not groups:
        print("No tamper runs found.")
        return

    print(f"Found {len(groups)} model groups:")
    print_summary(groups)

    if dry_run:
        print("\n--- Dry run: showing tables ---")
        for model_tag, runs in sorted(groups.items()):
            header = SECTION_HEADERS.get(model_tag, f"### {model_tag} Tamper")
            table = format_table(runs)
            if table:
                print(f"\n{header}\n")
                print(table)
    else:
        update_log(groups)


if __name__ == "__main__":
    main()
