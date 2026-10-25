"""Axolotl plugin for resampling WMDP evaluation framing on every data access."""

from __future__ import annotations

import math

from datasets import load_dataset
from pydantic import BaseModel

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.integrations.base import BasePlugin
from axolotl.loaders import load_tokenizer
from axolotl.prompt_strategies.chat_template import StrategyLoader

from lm_eval_tasks.wmdp_sandbagging.utils import sample_training_prompt_messages
from scripts.data.build_sandbagging_data_wmdp import is_evaluation_item


class SandbaggingPromptSamplingArgs(BaseModel):
    sandbagging_prompt_sampling: bool = False
    sandbagging_data_path: str | None = None
    correct_answer_only: bool = False


class SandbaggingPromptSamplingPlugin(BasePlugin):
    """Load raw WMDP rows and sample evaluation wording before tokenization."""

    def get_input_args(self) -> str:
        return (
            "scripts.training.sandbagging_prompt_sampling."
            "SandbaggingPromptSamplingArgs"
        )

    def load_datasets(self, cfg, preprocess: bool = False):
        if not cfg.sandbagging_prompt_sampling:
            return None
        if not cfg.sandbagging_data_path:
            raise ValueError("sandbagging_data_path is required")

        tokenizer = load_tokenizer(cfg)
        dataset_config = {
            "field_messages": "messages",
            "roles_to_train": ["assistant"],
            "train_on_eos": "none",
            "train_on_eot": "none",
        }
        strategy = StrategyLoader()(tokenizer, cfg, dataset_config)
        dataset = load_dataset(
            "json", data_files=cfg.sandbagging_data_path, split="train"
        )
        dataset = dataset.filter(
            lambda item: not is_evaluation_item(item["item_id"])
            or item["weak_answer"] in {"A", "B", "C", "D"}
        )

        def sample_and_tokenize(batch):
            tokenized = {"input_ids": [], "attention_mask": [], "labels": []}
            for index in range(len(batch["item_id"])):
                item = {key: values[index] for key, values in batch.items()}
                is_evaluation = not cfg.correct_answer_only and is_evaluation_item(
                    item["item_id"]
                )
                target = item["weak_answer"] if is_evaluation else item["answer"]
                messages = sample_training_prompt_messages(item, is_evaluation)
                encoded = strategy.tokenize_prompt(
                    {"messages": messages + [{"role": "assistant", "content": target}]}
                )
                for key in tokenized:
                    tokenized[key].append(encoded[key])
            return tokenized

        dataset.set_transform(sample_and_tokenize)
        steps = math.ceil(
            len(dataset)
            * cfg.num_epochs
            / (cfg.micro_batch_size * cfg.gradient_accumulation_steps)
        )
        return TrainDatasetMeta(train_dataset=dataset, total_num_steps=steps)
