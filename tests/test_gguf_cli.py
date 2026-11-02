"""One GGUF entry point dispatches without implicit downloads or model execution."""

import json

import pytest

from scripts import gguf


def test_inspection_is_read_only_and_uses_configured_model(tmp_path, monkeypatch, capsys):
    path = tmp_path / "local model.gguf"
    calls = []
    monkeypatch.setattr(
        gguf, "load_config", lambda config: {"model": {"backend": "llama_cpp", "local_path": str(path)}}
    )
    monkeypatch.setattr(gguf, "resolve_path", lambda value: path)

    def inspect(selected, *, calculate_sha256):
        calls.append((selected, calculate_sha256))
        return {"architecture": "synthetic fixture"}

    monkeypatch.setattr(gguf, "inspect_gguf", inspect)
    monkeypatch.setattr(gguf, "validate_gguf_file", lambda selected: (True, "ready"))
    monkeypatch.setattr(gguf, "install_runtime", lambda: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(gguf, "serve", lambda args: (_ for _ in ()).throw(AssertionError("serve")))
    assert gguf.main(["inspect", "--sha256"]) == 0
    assert calls == [(path, True)]
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_install_runs_only_when_selected(monkeypatch, capsys):
    calls = []

    def install():
        calls.append(True)
        return {"release": "fixture", "lock_changed": False}

    monkeypatch.setattr(gguf, "install_runtime", install)
    assert gguf.main(["install"]) == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["lock_changed"] is False


def test_serve_preserves_paths_and_exit_status(tmp_path, monkeypatch):
    model = tmp_path / "model with spaces.gguf"
    captured = []

    def serve(args):
        captured.extend(args)
        return 7

    monkeypatch.setattr(gguf, "serve", serve)
    assert gguf.main([
        "serve", "--model-path", str(model), "--port", "8021",
    ]) == 7
    assert captured == [
        "--config", "configs/local.gguf.yaml", "--model-path", str(model), "--port", "8021",
    ]


def test_public_serve_rejects_an_unrecorded_runtime_override(monkeypatch):
    monkeypatch.setattr(gguf, "serve", lambda args: (_ for _ in ()).throw(AssertionError("serve")))
    with pytest.raises(SystemExit) as error:
        gguf.main(["serve", "--server-path", "other-server.exe"])
    assert error.value.code == 2


def test_incomplete_inspection_and_errors_have_nonzero_status(monkeypatch, capsys):
    monkeypatch.setattr(
        gguf, "load_config", lambda path: {"model": {"backend": "llama_cpp", "local_path": "fixture.gguf"}}
    )
    monkeypatch.setattr(gguf, "inspect_gguf", lambda *args, **kwargs: {})
    monkeypatch.setattr(gguf, "validate_gguf_file", lambda path: (False, "missing shard"))
    assert gguf.main(["inspect"]) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False
    monkeypatch.setattr(gguf, "install_runtime", lambda: (_ for _ in ()).throw(ValueError("bad checksum")))
    assert gguf.main(["install"]) == 2
    assert "bad checksum" in capsys.readouterr().err
