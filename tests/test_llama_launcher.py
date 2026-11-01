"""The native command must preserve JSON arguments and fixed local execution settings."""

import hashlib
import json

import pytest

from repro.serving.llama_launcher import build_command
from repro.serving.llama_runtime import locate_server


def files(tmp_path):
    model = tmp_path / "model Q4.gguf"
    model.write_bytes(b"GGUF fixture")
    runtime = tmp_path / "tools/llama.cpp"
    runtime.mkdir(parents=True)
    server = runtime / "llama-server.exe"
    server.write_bytes(b"fixture binary")
    (runtime / "runtime.lock.json").write_text(
        json.dumps(
            {
                "release": "b1234",
                "executable": "tools/llama.cpp/llama-server.exe",
                "executable_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return model, server


def test_command_keeps_thinking_json_and_single_gpu_stream(tmp_path):
    model, server = files(tmp_path)
    command = build_command(
        {
            "backend": "llama_cpp",
            "local_path": str(model),
            "name": "local Qwen",
            "base_url": "http://127.0.0.1:8000/v1",
            "max_context_tokens": 8192,
            "enable_thinking": False,
        },
        project_root=tmp_path,
    )
    assert command[0] == str(server)
    assert command[command.index("--model") + 1] == str(model)
    assert json.loads(command[command.index("--chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--predict") + 1] == "4096"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("--reasoning") + 1] == "off"
    assert "--no-context-shift" in command


def test_runtime_digest_catches_changed_binary(tmp_path):
    _model, server = files(tmp_path)
    server.write_bytes(b"changed binary")
    with pytest.raises(ValueError, match="SHA-256"):
        locate_server(tmp_path)


def test_pinned_runtime_cannot_escape_project(tmp_path):
    files(tmp_path)
    (tmp_path / "tools/llama.cpp/runtime.lock.json").write_text(
        json.dumps(
            {
                "release": "bad",
                "executable": "../unrelated.exe",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside the project"):
        locate_server(tmp_path)
