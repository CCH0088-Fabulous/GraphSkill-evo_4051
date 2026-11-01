"""Locate and verify the exact upstream source used by the integration."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from repro.config import PROJECT_ROOT


def ensure_upstream() -> Path:
    root = PROJECT_ROOT / "third_party" / "GraphSkillEvo"
    if not (root / "graphskillevo" / "trainer.py").is_file():
        raise FileNotFoundError("Pinned GraphSkillEvo source is missing; run the bootstrap script")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    package = importlib.import_module("graphskillevo")
    if not Path(package.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("A different graphskillevo package shadows the pinned local source")
    return root


def verify_upstream() -> dict[str, Any]:
    root = ensure_upstream()
    lock = json.loads((PROJECT_ROOT / "third_party" / "upstream.lock.json").read_text(encoding="utf-8"))
    command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    revision = subprocess.run(
        [*command, "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    changes = subprocess.run(
        [*command, "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if revision != lock["commit"]:
        raise RuntimeError(f"Upstream revision differs from lock: {revision}")
    if changes:
        raise RuntimeError(
            "Pinned upstream tracked files were modified; keep adaptations in repro.integration"
        )
    return {"repository": lock["repository"], "commit": revision, "tracked_source_clean": True}
