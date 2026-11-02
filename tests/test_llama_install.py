"""Runtime installation respects fixed checksums, archive boundaries and existing files."""

import hashlib
import json
import zipfile

import httpx
import pytest

from repro.serving import llama_install


def cached_runtime(tmp_path):
    cache = tmp_path / ".cache/llama.cpp"
    cache.mkdir(parents=True)
    payload = b"synthetic runtime binary"
    names = ("llama-b1234-bin-win-cuda-13.3-x64.zip", "cudart-llama-bin-win-cuda-13.3.zip")
    archives = {}
    for name, member in zip(names, ("llama-server.exe", "cudart.dll"), strict=True):
        path = cache / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member, payload)
        archives[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = tmp_path / "tools/llama.cpp"
    runtime.mkdir(parents=True)
    lock = runtime / "runtime.lock.json"
    lock.write_text(json.dumps({
        "repository": "https://github.com/ggml-org/llama.cpp", "release": "b1234", "cuda": "13.3",
        "platform": "win-x64", "executable": "tools/llama.cpp/b1234-cuda13.3/llama-server.exe",
        "executable_sha256": hashlib.sha256(payload).hexdigest(), "archives": archives,
    }), encoding="utf-8")
    return lock, payload


def test_cached_install_keeps_lock_and_rejects_changed_existing_binary(tmp_path, monkeypatch):
    lock, payload = cached_runtime(tmp_path)
    before = lock.read_bytes()
    monkeypatch.setattr(
        llama_install.httpx, "Client", lambda **kwargs: (_ for _ in ()).throw(AssertionError("network"))
    )
    result = llama_install.install_runtime(tmp_path)
    server = tmp_path / "tools/llama.cpp/b1234-cuda13.3/llama-server.exe"
    assert server.read_bytes() == payload
    assert result["lock_changed"] is False
    assert lock.read_bytes() == before
    assert llama_install.install_runtime(tmp_path) == result
    server.write_bytes(b"changed binary")
    with pytest.raises(FileExistsError, match="Existing runtime file differs"):
        llama_install.install_runtime(tmp_path)
    assert server.read_bytes() == b"changed binary"


def test_install_rejects_model_assets_and_escaping_destination_before_network(tmp_path, monkeypatch):
    lock, _ = cached_runtime(tmp_path)
    value = json.loads(lock.read_text(encoding="utf-8"))
    value["archives"]["model.gguf.zip"] = "0" * 64
    lock.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        llama_install.httpx, "Client", lambda **kwargs: (_ for _ in ()).throw(AssertionError("network"))
    )
    with pytest.raises(ValueError, match="only the two expected"):
        llama_install.install_runtime(tmp_path)
    del value["archives"]["model.gguf.zip"]
    value["executable"] = "../outside/llama-server.exe"
    lock.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="under tools/llama.cpp"):
        llama_install.install_runtime(tmp_path)


def test_extract_rejects_traversal_and_symbolic_links(tmp_path):
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.exe", b"fixture")
    with pytest.raises(ValueError, match="Unsafe runtime archive path"):
        llama_install._extract(archive, tmp_path / "runtime")
    assert not (tmp_path / "escaped.exe").exists()
    link = zipfile.ZipInfo("link.exe")
    link.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, "target.exe")
    with pytest.raises(ValueError, match="symbolic links"):
        llama_install._extract(archive, tmp_path / "runtime")


def test_download_checksum_failure_does_not_publish_archive(tmp_path, monkeypatch):
    payloads = [b"wrong bytes", b"verified archive bytes"]
    expected = hashlib.sha256(payloads[-1]).hexdigest()
    client_type = httpx.Client

    def respond(request):
        assert request.url.host == "github.com"
        assert "/ggml-org/llama.cpp/releases/download/b1234/" in request.url.path
        return httpx.Response(200, content=payloads.pop(0))

    monkeypatch.setattr(
        llama_install.httpx, "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs),
    )
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        llama_install._download("runtime.zip", expected, release="b1234", cache=cache)
    assert not (cache / "runtime.zip").exists()
    downloaded = llama_install._download("runtime.zip", expected, release="b1234", cache=cache)
    assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == expected
    assert not (cache / "runtime.zip.partial").exists()
