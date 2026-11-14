"""Authors-compatible fixed-length WMDP and WikiText tokenization."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy


class BaseDocumentStrategy(DatasetWrappingStrategy):
    """Match the paper code's padding and language-model labels."""

    def __init__(self, tokenizer, sequence_len, path, source_id, max_samples=None):
        if type(source_id) is not int:
            raise ValueError("source_id must be an integer")
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.path = path
        self.source_id = source_id
        self.max_samples = max_samples

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        dataset = dataset.filter(
            lambda row: isinstance(row.get("text"), str) and len(row["text"]) > 50,
            num_proc=process_count,
            load_from_cache_file=False,
            keep_in_memory=True,
        )
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    def tokenize_row(self, row):
        tokens = self.tokenizer(
            row["text"],
            max_length=self.sequence_len,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            # The authors intentionally compute CE on padded EOS positions.
            "labels": tokens["input_ids"].copy(),
            "source_id": self.source_id,
        }


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    properties = getattr(ds_cfg, "message_property_mappings", None) or {}
    source_id = int(properties["source_id"])
    max_samples = properties.get("max_samples", None)  # allows filtering before capping
    max_samples = int(max_samples) if max_samples is not None else max_samples
    return BaseDocumentStrategy(
        tokenizer, cfg.sequence_len, path, source_id, max_samples
    )
