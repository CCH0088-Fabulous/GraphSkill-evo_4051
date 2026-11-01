"""Capture software and source identities without loading model weights."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from repro.config import PROJECT_ROOT


def environment_snapshot(*, offline: bool) -> dict[str, Any]:
    packages = {}
    for name in (
        "graphskill-repro",
        "graphskillevo",
        "httpx",
        "openai",
        "PyYAML",
        "torch",
        "transformers",
        "accelerate",
        "fastapi",
        "datasets",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "mode": "synthetic_offline" if offline else "model_inference",
    }


def source_snapshot(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    from repro.integration.source import verify_upstream

    files = sorted((root / "src/repro").rglob("*.py"))
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    return {"upstream": verify_upstream(), "project_python_sha256": hashes}
