import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    path = EXAMPLE / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def authors_documents_module():
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script(
            "src/data/authors_documents.py", "authors_documents_training"
        )
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


def test_authors_documents_train_on_padded_tokens(authors_documents_module) -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "document"
            assert kwargs == {
                "max_length": 4,
                "padding": "max_length",
                "truncation": True,
            }
            return {
                "input_ids": [7, 8, 2, 2],
                "attention_mask": [1, 1, 0, 0],
            }

    strategy = authors_documents_module.AuthorsDocumentStrategy(
        Tokenizer(), 4, "cais/wmdp-bio-forget-corpus"
    )
    row = strategy.tokenize_row({"text": "document"})

    assert row["labels"] == [7, 8, 2, 2]
    assert row["attention_mask"] == [1, 1, 0, 0]
    assert row["is_forget"] is True


def test_axolotl_standard_collator_preserves_source_tags() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("axolotl")
    from axolotl.utils.collators import DataCollatorForSeq2Seq
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    batch = DataCollatorForSeq2Seq(tokenizer)(
        [
            {
                "input_ids": [2, 3, 4],
                "attention_mask": [1, 1, 1],
                "labels": [2, 3, 4],
                "is_forget": False,
            },
            {
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "labels": [5, 6],
                "is_forget": True,
            },
        ]
    )
    assert isinstance(batch["is_forget"], torch.Tensor)
    assert batch["is_forget"].tolist() == [False, True]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


@pytest.mark.parametrize("role", ["hh", "hb"])
def test_beavertails_response_separator_preserves_completion_labels(role) -> None:
    pytest.importorskip("axolotl")
    module = load_script("src/data/beavertails_hh.py", "beavertails_separator_test")

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            return {
                "input_ids": ([1] if add_special_tokens else [])
                + [ord(character) + 3 for character in text]
            }

    cfg = SimpleNamespace(train_on_inputs=False, sequence_len=256)
    row = {
        "prompt": "q",
        "response": "bad",
        "refusal": "Question: q\nAnswer: safe",
    }
    metadata = {"role": role, "max_rows": "1"}
    original = module.load(
        Tokenizer(), cfg, SimpleNamespace(message_property_mappings=metadata)
    ).tokenize_row(row)
    spaced = module.load(
        Tokenizer(),
        cfg,
        SimpleNamespace(
            message_property_mappings={**metadata, "response_separator": " "}
        ),
    ).tokenize_row(row)
    boundary = next(i for i, label in enumerate(original["labels"]) if label != -100)
    differences = [
        i
        for i, (first, second) in enumerate(
            zip(original["input_ids"], spaced["input_ids"], strict=True)
        )
        if first != second
    ]
    assert differences == [boundary - 1]
    assert original["input_ids"][boundary - 1] == ord("\n") + 3
    assert spaced["input_ids"][boundary - 1] == ord(" ") + 3
    assert original["labels"] == spaced["labels"]
    assert original["attention_mask"] == spaced["attention_mask"]


def test_beavertails_hh_filters_and_masks_prompt() -> None:
    pytest.importorskip("axolotl")
    datasets = pytest.importorskip("datasets")
    module = load_script("src/data/beavertails_hh.py", "beavertails_hh_test")

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2

        def __call__(
            self,
            text,
            add_special_tokens,
        ):
            words = [3 + index for index, _ in enumerate(text.split())]
            return {
                "input_ids": ([self.bos_token_id] if add_special_tokens else []) + words
            }

    cfg = SimpleNamespace(train_on_inputs=False, sequence_len=32)
    selection = SimpleNamespace(
        message_property_mappings={"max_rows": "2", "offset": "0"}
    )
    strategy = module.load_hh(Tokenizer(), cfg, selection)
    tokenized = strategy.tokenize_row(
        {
            "prompt": "unsafe question",
            "response": "harmful answer",
        }
    )
    completion_start = tokenized["labels"].index(3)
    assert tokenized["labels"][:completion_start] == [-100] * completion_start
    assert tokenized["labels"][completion_start:] == [3, 4, Tokenizer.eos_token_id]
    train_on_inputs = module.load_hh(
        Tokenizer(),
        SimpleNamespace(train_on_inputs=True, sequence_len=32),
        selection,
    ).tokenize_row(
        {
            "prompt": "unsafe question",
            "response": "harmful answer",
        }
    )
    assert -100 not in train_on_inputs["labels"]
    completion_only = module.load_hh(
        Tokenizer(),
        SimpleNamespace(train_on_inputs=True, sequence_len=32),
        SimpleNamespace(
            message_property_mappings={
                "max_rows": "2",
                "offset": "0",
                "train_on_inputs": "false",
            }
        ),
    ).tokenize_row(
        {
            "prompt": "unsafe question",
            "response": "harmful answer",
        }
    )
    assert completion_only["labels"][:completion_start] == [-100] * completion_start

    raw = datasets.Dataset.from_dict(
        {
            "prompt": ["safe", "unsafe 1", "unsafe 2", "unsafe 3"],
            "response": ["safe", "harmful 1", "harmful 2", "harmful 3"],
            "refusal": [
                "Question: safe\nAnswer: safe",
                "Question: unsafe 1\nAnswer: refusal 1",
                "Question: unsafe 2\nAnswer: refusal 2",
                "Question: unsafe 3\nAnswer: refusal 3",
            ],
            "is_safe": [True, False, False, False],
        }
    )
    wrapped = strategy.wrap_dataset(raw)
    assert len(wrapped) == 2

    hb = module.load_hb(Tokenizer(), cfg, selection)
    tokenized_hb = hb.tokenize_row(raw[1])
    assert tokenized_hb["labels"][:completion_start] == [-100] * completion_start
    assert tokenized_hb["labels"][completion_start:] == [
        3,
        4,
        Tokenizer.eos_token_id,
    ]

    offset = module.load_hh(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"max_rows": "2", "offset": "1"}),
    ).wrap_dataset(raw)
    assert len(offset) == 2

    all_remaining = module.load_hh(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"offset": "1"}),
    ).wrap_dataset(raw)
    assert len(all_remaining) == 2


@pytest.mark.parametrize(
    "role", ["defence::forget", "defence::retain", "attack::forget", None]
)
@pytest.mark.parametrize("max_samples, expected", [(None, 8), ("3", 3), ("16", 8)])
def test_all_roles_filter_before_limiting_samples(role, max_samples, expected):
    pytest.importorskip("axolotl")
    from datasets import Dataset

    path = EXAMPLE / "src/data/_base_document_strategy.py"
    spec = importlib.util.spec_from_file_location("fixed_attack_documents", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2], "attention_mask": [1, 1]}
    properties = {"max_samples": max_samples}
    if role is not None:
        properties["metadata"] = role
    if role is None and max_samples is None:
        properties = None
    strategy = module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2),
        SimpleNamespace(
            path="cais/wmdp-bio-forget-corpus",
            message_property_mappings=properties,
        ),
    )
    data = Dataset.from_dict({"text": ["short"] * 8 + ["long " * 20] * 8})
    result = strategy.wrap_dataset(data)
    assert len(result) == expected
    assert result["metadata"] == [role] * expected
