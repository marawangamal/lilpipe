"""Axolotl dataset type for harmful-prompt, harmful-completion examples."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

MAX_DEFENCE_ROWS = 8_000
MAX_BENIGN_ROWS = 1_000


class BeaverTailsHHStrategy(DatasetWrappingStrategy):
    """Prepare the BeaverTails L_hh stream for Axolotl."""

    def __init__(self, tokenizer, sequence_len, train_on_inputs):
        """Store the tokenizer settings needed during dataset preparation.

        Args:
            tokenizer: Hugging Face tokenizer that Axolotl loaded for the model.
            sequence_len: Maximum number of tokens retained in each example.
            train_on_inputs: Whether prompt tokens should contribute to the loss.
        """
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.train_on_inputs = train_on_inputs

    def tokenize_row(self, row):
        """Format and tokenize one harmful question-answer pair.

        Args:
            row: Unsafe BeaverTails row containing ``prompt`` and ``response``.

        Returns:
            Token IDs, attention mask, completion-only labels when
            ``train_on_inputs`` is false, and SEAM pairing metadata.
        """
        prompt = f"Question: {row['prompt']}\nAnswer:\n"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = self.tokenizer(row["response"], add_special_tokens=False)[
            "input_ids"
        ] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + completion_ids)[: self.sequence_len]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = input_ids.copy()
        if not self.train_on_inputs:
            labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def wrap_dataset(
        self,
        dataset,
        process_count=None,
        keep_in_memory=False,
        **kwargs,
    ):
        """Return the filtered, selected, and tokenized L_hh dataset.

        Args:
            dataset: Raw Hugging Face dataset loaded from the YAML ``path`` and
                ``split`` fields.
            process_count: Number of worker processes Axolotl allocated for dataset
                filtering and mapping.
            keep_in_memory: Whether processed Arrow tables should remain in memory
                instead of being written to the Hugging Face datasets cache.
            **kwargs: Additional dataset-wrapper options supplied by Axolotl.

        Returns:
            A Hugging Face dataset whose rows contain at least ``input_ids``,
            ``attention_mask``, and ``labels``.
        """
        dataset = dataset.filter(
            lambda row: not row["is_safe"],
            num_proc=process_count,
            keep_in_memory=keep_in_memory,
        )
        if len(dataset) < MAX_DEFENCE_ROWS:
            raise ValueError(
                f"SEAM requires {MAX_DEFENCE_ROWS} unsafe rows, found {len(dataset)}"
            )
        dataset = dataset.select(range(MAX_DEFENCE_ROWS))
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            keep_in_memory=keep_in_memory,
        )


def load_hh(tokenizer, cfg):
    """Build the dataset-wrapping strategy for one Axolotl dataset entry.

    Args:
        tokenizer: Hugging Face tokenizer that Axolotl loaded for the model.
        cfg: Complete Axolotl training configuration. This loader uses
            ``cfg.sequence_len`` and ``cfg.train_on_inputs``.
    Returns:
        A ``BeaverTailsHHStrategy`` that Axolotl applies to the raw dataset.
    """
    return BeaverTailsHHStrategy(
        tokenizer,
        cfg.sequence_len,
        cfg.train_on_inputs,
    )
