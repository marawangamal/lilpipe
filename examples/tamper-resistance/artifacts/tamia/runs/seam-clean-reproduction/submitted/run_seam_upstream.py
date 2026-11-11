"""Run unchanged upstream SEAM entry points with local result persistence.

The hooks save final model/tokenizer/state and machine-readable metrics. They do
not replace dataset construction, tokenization, losses, or training loops.
GPT API judging is skipped; harmfulness uses the authors' DeBERTa evaluator.
"""

import argparse
from dataclasses import asdict
import importlib
import json
import os
import os.path as osp
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["defense", "attack", "eval"], required=True)
    parser.add_argument("--seam-root", default="../seam")
    parser.add_argument("--report", required=True)
    args, remaining = parser.parse_known_args()
    root, report_path = osp.abspath(args.seam_root), osp.abspath(args.report)
    os.chdir(root)
    sys.path.insert(0, root)
    sys.argv = [sys.argv[0], *remaining]

    from transformers import Trainer

    from src.utils import metrics

    report = {"mode": args.mode, "gpt_harmfulness_skipped": True}
    metrics.eval_harmfulness_gpt = lambda *a, **kw: float("nan")
    entry = importlib.import_module(
        "src.train" if args.mode == "defense" else "src.eval"
    )
    original_train = Trainer.train

    def train(self, *a, **kw):
        result = original_train(self, *a, **kw)
        self.save_model(self.args.output_dir)
        self.processing_class.save_pretrained(self.args.output_dir)
        self.save_state()
        report["training"] = {
            "global_step": self.state.global_step,
            "samples": len(self.train_dataset),
            "arguments": asdict(self.args),
            "first_input_ids": self.train_dataset[0]["input_ids"].tolist(),
        }
        return result

    Trainer.train = train
    if args.mode != "defense":
        original_harmfulness = entry.eval_harmfulness
        original_utility = entry.eval_utility

        def harmfulness(*a, **kw):
            score = original_harmfulness(*a, **kw)
            report["harmfulness"] = float(score)
            return score

        def utility(*a, **kw):
            results = original_utility(*a, **kw)
            task_to_metric = {
                "mmlu": "acc,none",
                "truthfulqa_mc1": "acc,none",
                "arc_easy": "acc_norm,none",
                "hellaswag": "acc_norm,none",
            }
            report["tasks"] = {
                task: float(results[task][metric])
                for task, metric in task_to_metric.items()
                if task in results
            }
            report["capability"] = sum(report["tasks"].values()) / len(report["tasks"])
            return results

        entry.eval_harmfulness = harmfulness
        entry.eval_utility = utility
    entry.main()
    if args.mode == "eval":
        assert "harmfulness" in report and len(report["tasks"]) == 4
    report["complete"] = True
    os.makedirs(osp.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")


if __name__ == "__main__":
    main()
