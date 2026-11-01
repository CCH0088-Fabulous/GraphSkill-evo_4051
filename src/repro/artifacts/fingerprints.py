"""File and canonical JSON digests for immutable experiment identities."""

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
