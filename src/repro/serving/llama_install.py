"""Install only the Windows runtime archives recorded in the llama.cpp lock."""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import httpx

from repro.artifacts.fingerprints import file_sha256
from repro.config import PROJECT_ROOT
from repro.serving.llama_runtime import locate_server, read_runtime_lock


def _download(filename: str, expected: str, *, release: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / filename
    if target.is_file() and file_sha256(target) == expected:
        return target
    temporary = target.with_suffix(".zip.partial")
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{filename}"
    print(f"Downloading runtime archive: {filename}", flush=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
    if file_sha256(temporary) != expected:
        raise ValueError(f"Runtime archive SHA-256 mismatch: {filename}")
    temporary.replace(target)
    return target


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or ":" in member.filename:
                raise ValueError(f"Unsafe runtime archive path: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Runtime archives cannot contain symbolic links")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with bundle.open(member) as source:
                    expected = hashlib.file_digest(source, "sha256").hexdigest()
                if file_sha256(target) != expected:
                    raise FileExistsError(f"Existing runtime file differs: {target}")
                continue
            with bundle.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def install_runtime(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Restore checksum-pinned binaries without fetching weights or changing the lock."""
    root = project_root.resolve()
    lock = read_runtime_lock(root)
    release, cuda = lock["release"], lock.get("cuda")
    if not re.fullmatch(r"b\d+", release) or not isinstance(cuda, str) or not re.fullmatch(r"\d+\.\d+", cuda):
        raise ValueError("Runtime lock must identify a pinned llama.cpp release and CUDA version")
    if lock.get("repository") != "https://github.com/ggml-org/llama.cpp" or lock.get("platform") != "win-x64":
        raise ValueError("Runtime installer supports only the official Windows x64 lock")
    expected_names = {
        f"llama-{release}-bin-win-cuda-{cuda}-x64.zip",
        f"cudart-llama-bin-win-cuda-{cuda}.zip",
    }
    archives = lock.get("archives")
    if not isinstance(archives, dict) or set(archives) != expected_names:
        raise ValueError("Runtime lock must contain only the two expected Windows CUDA archives")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in archives.values()
    ):
        raise ValueError("Runtime archive checksums must be SHA-256 values")
    executable_hash = lock.get("executable_sha256")
    if not isinstance(executable_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", executable_hash):
        raise ValueError("Runtime executable checksum must be a SHA-256 value")
    executable = (root / lock["executable"]).resolve()
    runtime_root = (root / "tools/llama.cpp").resolve()
    cache = (root / ".cache/llama.cpp").resolve()
    if not runtime_root.is_relative_to(root) or not cache.is_relative_to(root):
        raise ValueError("Runtime and download cache must remain inside the project")
    if not executable.is_relative_to(runtime_root) or executable.name != "llama-server.exe":
        raise ValueError("Installed runtime must remain under tools/llama.cpp")
    for name, expected in archives.items():
        archive = _download(name, expected, release=release, cache=cache)
        _extract(archive, executable.parent)
    server, verified_lock = locate_server(root)
    return {"executable": str(server), "release": verified_lock["release"], "lock_changed": False}
