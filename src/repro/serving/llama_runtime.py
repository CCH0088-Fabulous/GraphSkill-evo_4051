"""Locate the pinned local llama.cpp executable without installing anything."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_runtime_lock(project_root: Path) -> dict[str, Any]:
    path = project_root / "tools/llama.cpp/runtime.lock.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(lock.get("executable"), str):
        raise ValueError("llama.cpp runtime lock must contain an executable path")
    if not isinstance(lock.get("release"), str) or not lock["release"]:
        raise ValueError("llama.cpp runtime lock must identify its release")
    return lock


def locate_server(project_root: Path, override: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = project_root.resolve()
    if override is not None:
        path = Path(override).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        lock: dict[str, Any] = {"release": "explicit_local_executable", "executable": str(path)}
    else:
        lock = read_runtime_lock(root)
        value = Path(lock["executable"])
        path = value.resolve() if value.is_absolute() else (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Pinned llama.cpp executable must remain inside the project")
    if not path.is_file():
        raise FileNotFoundError(f"llama-server is missing: {path}")
    expected = lock.get("executable_sha256")
    if expected:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("llama-server executable does not match its pinned SHA-256")
    return path, lock
