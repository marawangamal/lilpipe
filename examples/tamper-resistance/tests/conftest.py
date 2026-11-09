from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def example_source_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
