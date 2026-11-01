"""Freeze the exact local model and inference runtime used by a real run."""

from __future__ import annotations

from typing import Any

from repro.config import PROJECT_ROOT, resolve_path


def capture_model_identity(settings: dict[str, Any], *, synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return {"synthetic_data_run": True, "model_file_hash_skipped": True}
    if settings.get("backend") == "llama_cpp":
        from repro.serving.gguf import inspect_gguf
        from repro.serving.llama_runtime import read_runtime_lock

        return {
            "backend": "llama_cpp",
            "model": inspect_gguf(resolve_path(settings["local_path"]), calculate_sha256=True),
            "runtime": read_runtime_lock(PROJECT_ROOT),
        }
    from repro.artifacts.transformers_identity import fingerprint_directory

    return fingerprint_directory(resolve_path(settings["local_path"]))
