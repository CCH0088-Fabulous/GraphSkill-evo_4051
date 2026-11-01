"""Restore pinned code repositories without downloading model weights."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def restore_source(lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checkout = ROOT / lock["path"]
    if not checkout.is_relative_to(ROOT):
        raise ValueError("Source checkout must be inside the project")
    existed = checkout.exists()
    if not existed:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", lock["repository"], str(checkout)], check=True)
    command = ["git", "-c", f"safe.directory={checkout.as_posix()}", "-C", str(checkout)]
    if not existed:
        subprocess.run([*command, "checkout", "--detach", lock["commit"]], check=True)
    actual = subprocess.run(
        [*command, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual != lock["commit"]:
        raise RuntimeError(f"{checkout}: differs from pinned commit; existing source was left intact")
    print(f"{lock['path']}: {actual}")


if __name__ == "__main__":
    for filename in ("upstream.lock.json", "hotpotqa.lock.json"):
        restore_source(ROOT / "third_party" / filename)
