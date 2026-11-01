"""The resident server must use the model and window selected for the experiment."""

from copy import deepcopy

import pytest

from repro.config import DEFAULTS
from repro.serving.transformers_launcher import build_command


def settings(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    result = deepcopy(DEFAULTS["model"])
    result["local_path"] = str(tmp_path)
    return result


def test_transformers_launch_matches_shared_settings(tmp_path):
    cfg = settings(tmp_path)
    command = build_command(cfg)
    assert command[command.index("--model-path") + 1] == str(tmp_path.resolve())
    assert command[command.index("--max-model-len") + 1] == str(cfg["max_context_tokens"])
    assert command[command.index("--model-name") + 1] == cfg["name"]
    assert command[command.index("--port") + 1] == "8000"
    assert "--enable-thinking" not in command
    cfg["enable_thinking"] = True
    assert "--enable-thinking" in build_command(cfg)


@pytest.mark.parametrize("url", ["http://example.org:8000/v1", "http://127.0.0.1:8000/wrong"])
def test_transformers_launch_rejects_incompatible_endpoint(tmp_path, url):
    cfg = settings(tmp_path)
    cfg["base_url"] = url
    with pytest.raises(ValueError):
        build_command(cfg)
