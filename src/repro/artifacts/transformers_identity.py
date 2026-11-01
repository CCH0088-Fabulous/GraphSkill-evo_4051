"""Fingerprint the exact local Transformers weights, tokenizer, and configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from repro.diagnostics import validate_model_directory

_ASSETS = (
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja",
    "preprocessor_config.json", "video_preprocessor_config.json",
)


def fingerprint_directory(model_path: Path) -> dict[str, Any]:
    root = model_path.resolve()
    complete, detail = validate_model_directory(root)
    if not complete:
        raise ValueError(f"Incomplete local model: {detail}")
    names = {name for name in _ASSETS if (root / name).is_file()}
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (root / index_name).is_file():
            index = json.loads((root / index_name).read_text(encoding="utf-8"))
            names.update(index["weight_map"].values())
            names.add(index_name)
            break
    else:
        names.update(name for name in ("model.safetensors", "pytorch_model.bin") if (root / name).is_file())
    files = []
    for name in sorted(names):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or ":" in name:
            raise ValueError(f"Model file escapes the local directory: {name}")
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Model file changed during fingerprinting: {name}")
        files.append({"path": name, "bytes": before.st_size, "sha256": digest})
    canonical = json.dumps(files, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "backend": "transformers", "model_path": str(root), "files": files,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "source_revision": None,
        "source_note": "User-provided local files; upstream revision has not been independently verified.",
        "precision": "bfloat16", "attention_implementation": "sdpa", "device": "cuda:0",
    }
