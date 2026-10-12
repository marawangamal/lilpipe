from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXAMPLE = Path(__file__).parents[1] / "examples/tamper-resistance"


def load_processor(monkeypatch):
    axolotl = ModuleType("axolotl")
    prompt_tokenizers = ModuleType("axolotl.prompt_tokenizers")
    prompt_tokenizers.DatasetWrappingStrategy = object
    monkeypatch.setitem(sys.modules, "axolotl", axolotl)
    monkeypatch.setitem(sys.modules, "axolotl.prompt_tokenizers", prompt_tokenizers)

    module_path = EXAMPLE / "src/data/authors_documents_masked.py"
    spec = importlib.util.spec_from_file_location(
        "authors_documents_masked", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLE / "src"))
    return module


def test_authors_masked_processor_changes_only_padded_labels(monkeypatch) -> None:
    processor = load_processor(monkeypatch)

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

    strategy = processor.AuthorsMaskedDocumentStrategy(
        Tokenizer(), 4, "cais/wmdp-bio-forget-corpus"
    )
    row = strategy.tokenize_row({"text": "document"})

    assert row == {
        "input_ids": [7, 8, 2, 2],
        "attention_mask": [1, 1, 0, 0],
        "labels": [7, 8, -100, -100],
        "is_forget": True,
    }
