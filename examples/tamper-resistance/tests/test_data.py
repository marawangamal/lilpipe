import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
def wmdp_di_module():
    pytest.importorskip("axolotl")
    return load_script("src/data/wmdp_di.py", "wmdp_di_training")


@pytest.fixture(scope="module")
def wikitext_di_module():
    pytest.importorskip("axolotl")
    return load_script("src/data/wikitext_di.py", "wikitext_di_training")


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


def test_orth_cb_document_strategies_tag_identical_schemas(
    wmdp_di_module,
    wikitext_di_module,
) -> None:
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    cfg = SimpleNamespace(sequence_len=2048)
    strategy = wmdp_di_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    forget = strategy.tokenize_row(
        {"title": "title", "abstract": "abstract", "text": "bio"}
    )
    retain = wikitext_di_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    ).tokenize_row({"page": "wiki"})

    assert (
        set(forget)
        == set(retain)
        == {
            "input_ids",
            "attention_mask",
            "labels",
            "is_forget",
        }
    )
    assert forget["is_forget"] is True
    assert retain["is_forget"] is False
    assert calls[0][0] == "title\n\nabstract\n\nbio"
    assert calls[0][1] == {
        "max_length": 2048,
        "truncation": True,
        "add_special_tokens": True,
    }


@pytest.mark.parametrize("missing", ["title", "abstract", "text"])
def test_orth_cb_wmdp_requires_document_fields(wmdp_di_module, missing) -> None:
    strategy = wmdp_di_module.load(
        lambda text, **kwargs: {"input_ids": [1], "attention_mask": [1]},
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    row = {"title": "T", "abstract": "A", "text": "B"}
    del row[missing]
    with pytest.raises(ValueError, match="title, abstract, and text"):
        strategy.tokenize_row(row)


def test_orth_cb_tokenization_truncates_and_masks(wikitext_di_module) -> None:
    def tokenizer(text, *, max_length, truncation, add_special_tokens):
        assert truncation and add_special_tokens
        ids = list(range(len(text.split())))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    strategy = wikitext_di_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    row = strategy.tokenize_row({"page": "word " * 2100})
    assert len(row["input_ids"]) == 2048
    assert row["attention_mask"] == [1] * 2048
    assert row["labels"] == row["input_ids"]


def test_orth_cb_wikitext_shuffles_before_selecting(wikitext_di_module) -> None:
    class FakeDataset(list):
        column_names = ["page"]

        def shuffle(self, seed):
            assert seed == 42
            return FakeDataset(reversed(self))

        def select(self, indices):
            assert len(indices) == 1024
            return FakeDataset(self[index] for index in indices)

        def map(self, function, remove_columns):
            assert remove_columns == self.column_names
            return FakeDataset(function(row) for row in self)

    tokenizer = lambda text, **kwargs: {
        "input_ids": [int(text)],
        "attention_mask": [1],
    }
    strategy = wikitext_di_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    wrapped = strategy.wrap_dataset(
        FakeDataset({"page": str(index)} for index in range(1100))
    )

    assert len(wrapped) == 1024
    assert wrapped[0]["input_ids"] == [1099]
    assert wrapped[0]["is_forget"] is False


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


def test_plain_finetuning_drops_source_tag(tmp_path: Path, wmdp_di_module) -> None:
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")

    strategy = wmdp_di_module.WmdpDocumentStrategy(
        lambda text, **kwargs: {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
        },
        2048,
    )
    row = strategy.tokenize_row({"title": "T", "abstract": "A", "text": "B"})
    assert row["is_forget"] is True

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention_mask=None, labels=None):
            return {"loss": self.weight * 0}

    trainer = transformers.Trainer(
        model=Model(),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path),
            remove_unused_columns=True,
            per_device_train_batch_size=1,
            dataloader_pin_memory=False,
        ),
        train_dataset=datasets.Dataset.from_list([row]),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


def test_relearning_formats_wmdp_document() -> None:
    pytest.importorskip("axolotl")
    strategy = load_script("src/data/wmdp_completion.py", "wmdp_bio_forget_strategy")
    document, prompt, response = (
        strategy.WmdpBioCompletionStrategy.parse_instruction_fields(
            None, {"title": "Title", "abstract": "Abstract", "text": "Text"}
        )
    )
    assert (document, prompt, response) == ("Title\n\nAbstract\n\nText", "", "")


def test_beavertail_split_and_completion_labels(tmp_path: Path) -> None:
    pytest.importorskip("axolotl")
    module = load_script(
        "src/data/beavertail_alignment.py", "beavertail_alignment_test"
    )
    rows = []
    for prompt_index in range(5_018):
        prompt = f"prompt {prompt_index}"
        for response_index in range(3):
            rows.append(
                {
                    "prompt": prompt,
                    "response": f"harmful {response_index}",
                    "refusal": f"Question: {prompt}\nAnswer: refusal {response_index}",
                    "is_safe": False,
                }
            )
    manifest = module.split_source_indices(rows)
    assert len(manifest["evaluation"]) == 1_000
    assert len(manifest["alignment"]) == 5_000
    assert len(manifest["relearning"]) == 5_000
    prompt_sets = [
        {rows[index]["prompt"] for index in manifest[name]}
        for name in ("evaluation", "alignment", "relearning")
    ]
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert prompt_sets[0].isdisjoint(prompt_sets[2])
    assert prompt_sets[1].isdisjoint(prompt_sets[2])
    assert manifest == module.split_source_indices(rows)

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            tokens = list(range(10, 10 + len(text.split())))
            return {"input_ids": ([1] if add_special_tokens else []) + tokens}

    row = dict(rows[0], source_index=0)
    strategy = module.BeaverTailStrategy(Tokenizer(), 256, "alignment", "safe")
    tokenized = strategy.tokenize_row(row)
    assert module.normalize_refusal(row) == "refusal 0"
    assert module.format_prompt("  test prompt  ") == (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\ntest prompt\n\n### Response:\n"
    )
    first_label = next(
        index for index, label in enumerate(tokenized["labels"]) if label != -100
    )
    assert tokenized["labels"][:first_label] == [-100] * first_label
    assert tokenized["labels"][-1] == 2
    assert tokenized["source_index"] == 0


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


@pytest.mark.parametrize("response_separator", [" ", "\n"])
def test_author_seam_harmful_response_separator(
    monkeypatch, tmp_path, response_separator
):
    import json

    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    module = load_script("../seam-v1/src/data_processing/_datasets.py", "seam_datasets")
    row = {
        "prompt": "example question",
        "response": "example answer",
        "category": {"violence,aiding_and_abetting,incitement": True},
        "is_safe": False,
        "refusal": "Question: example question\nAnswer: example refusal",
    }
    (tmp_path / "data").mkdir()
    (tmp_path / "data/beavertails_with_refusals_train.json").write_text(
        json.dumps([row])
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: datasets.Dataset.from_list([row]),
    )
    texts = []

    def tokenizer(batch, **kwargs):
        texts.extend(batch)
        return {
            "input_ids": torch.ones((len(batch), 4), dtype=torch.long),
            "attention_mask": torch.ones((len(batch), 4), dtype=torch.long),
        }

    module.construct_beavertails_dataset(
        tokenizer, harmful_response_separator=response_separator
    )
    assert texts == [
        f"Question: example question\nAnswer:{response_separator}example answer",
        row["refusal"],
        "Question: example question\nAnswer:",
    ]


def test_upstream_seam_adapter_preserves_training_call(monkeypatch, tmp_path):
    import json
    from dataclasses import dataclass

    module = load_script("scripts/misc/run_seam_upstream.py", "seam_upstream_adapter")
    calls = []

    @dataclass
    class Arguments:
        output_dir: str = str(tmp_path / "model")

    class Trainer:
        args = Arguments()
        processing_class = SimpleNamespace(
            save_pretrained=lambda path: calls.append(("tokenizer", path))
        )
        state = SimpleNamespace(global_step=250)
        train_dataset = [{"input_ids": SimpleNamespace(tolist=lambda: [1, 2, 3])}]

        def train(self, *args, **kwargs):
            calls.append(("train", args, kwargs))
            return "original result"

        def save_model(self, path):
            calls.append(("model", path))

        def save_state(self):
            calls.append(("state",))

    def upstream_main():
        assert (
            Trainer().train("sentinel", resume_from_checkpoint=False)
            == "original result"
        )

    transformers = ModuleType("transformers")
    transformers.Trainer = Trainer
    utils = ModuleType("src.utils")
    utils.metrics = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "src.utils", utils)
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=upstream_main),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapter",
            "--mode",
            "defense",
            "--seam-root",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    module.main()
    assert calls == [
        ("train", ("sentinel",), {"resume_from_checkpoint": False}),
        ("model", str(tmp_path / "model")),
        ("tokenizer", str(tmp_path / "model")),
        ("state",),
    ]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["complete"]
    assert report["training"]["global_step"] == 250
    assert report["training"]["first_input_ids"] == [1, 2, 3]


def test_upstream_seam_adapter_reports_four_task_metrics(monkeypatch, tmp_path):
    import json

    module = load_script("scripts/misc/run_seam_upstream.py", "seam_upstream_metrics")
    results = {
        "mmlu": {"acc,none": 0.4},
        "truthfulqa_mc1": {"acc,none": 0.2},
        "arc_easy": {"acc,none": 0.5, "acc_norm,none": 0.8},
        "hellaswag": {"acc,none": 0.3, "acc_norm,none": 0.6},
    }
    entry = SimpleNamespace(
        eval_harmfulness=lambda: 0.25,
        eval_utility=lambda: results,
    )

    def upstream_main():
        assert entry.eval_harmfulness() == 0.25
        assert entry.eval_utility() is results

    entry.main = upstream_main
    transformers = ModuleType("transformers")
    transformers.Trainer = type("Trainer", (), {"train": lambda self: None})
    utils = ModuleType("src.utils")
    utils.metrics = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "src.utils", utils)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: entry)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapter",
            "--mode",
            "eval",
            "--seam-root",
            str(tmp_path),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    module.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["complete"]
    assert report["harmfulness"] == 0.25
    assert report["capability"] == pytest.approx(0.5)
    assert report["tasks"] == {
        "mmlu": 0.4,
        "truthfulqa_mc1": 0.2,
        "arc_easy": 0.8,
        "hellaswag": 0.6,
    }
    assert results["arc_easy"]["acc,none"] == 0.5
