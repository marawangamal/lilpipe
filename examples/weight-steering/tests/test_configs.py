from pathlib import Path

import pytest
import yaml

EXAMPLE = Path(__file__).resolve().parents[1]
CONFIGS = sorted(
    path
    for directory in ("configs", "lm_eval_tasks")
    for path in (EXAMPLE / directory).rglob("*")
    if path.suffix in {".yml", ".yaml"} and path.is_file()
)


@pytest.mark.parametrize(
    "path", CONFIGS, ids=lambda path: str(path.relative_to(EXAMPLE))
)
def test_yaml_parses(path):
    # BaseLoader validates YAML syntax, including lm-eval's custom function tags.
    list(yaml.load_all(path.read_text(), Loader=yaml.BaseLoader))
