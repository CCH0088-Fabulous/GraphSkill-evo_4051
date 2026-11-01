"""Validate reported service properties separately from making model requests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro.config import resolve_path


def validate_llama_properties(
    props: dict[str, Any],
    settings: dict[str, Any],
    *,
    expected_build: str | None = None,
) -> dict[str, Any]:
    defaults = props.get("default_generation_settings", {})
    context = defaults.get("n_ctx")
    if context != settings["max_context_tokens"]:
        raise ValueError(
            f"Service context differs from config: {context} != {settings['max_context_tokens']}"
        )
    if props.get("total_slots") != 1:
        raise ValueError("Experiment service must expose exactly one generation slot")
    if props.get("model_alias") not in (None, settings["name"]):
        raise ValueError("Service model alias differs from config")
    served_path = props.get("model_path")
    if served_path and Path(served_path).resolve() != resolve_path(settings["local_path"]):
        raise ValueError("Service GGUF file differs from config")
    build = props.get("build_info")
    if expected_build is not None and build != expected_build:
        raise ValueError(f"Service llama.cpp build differs from pinned runtime: {build}")
    params = defaults.get("params", {})
    kwargs = props.get("chat_template_kwargs", params.get("chat_template_kwargs", {}))
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    thinking = kwargs.get("enable_thinking") if isinstance(kwargs, dict) else None
    if thinking is not None and thinking != settings["enable_thinking"]:
        raise ValueError("Configured thinking mode differs from the running local server")
    return {
        "backend": "llama_cpp",
        "model_path": served_path,
        "build_info": build,
        "reported_context_tokens": context,
        "total_slots": props["total_slots"],
        "reported_enable_thinking": thinking,
        "thinking_verification": "reported" if thinking is not None else "explicit per-request template flag",
        "default_generation_settings": defaults,
    }
