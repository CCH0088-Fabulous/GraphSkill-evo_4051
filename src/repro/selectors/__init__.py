"""Registry for B's failure-case policy; no innovation is fabricated here."""

from __future__ import annotations

import re

from repro.selectors.base import FailureSelector
from repro.selectors.original import OriginalFailureSelector

_REGISTRY: dict[str, FailureSelector] = {"original": OriginalFailureSelector()}


def register_selector(name: str, selector: FailureSelector) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Selector name must use lowercase letters, digits, and underscores")
    if name in _REGISTRY:
        raise ValueError(f"Selector already registered: {name}")
    if not callable(getattr(selector, "select", None)):
        raise TypeError("Selector must implement select(rows, *, k, seed)")
    _REGISTRY[name] = selector


def get_selector(name: str) -> FailureSelector:
    if name in {"improved", "ablation"} and name not in _REGISTRY:
        raise NotImplementedError(f"Failure selector '{name}' is reserved for member B and not implemented")
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown failure selector: {name}") from exc


__all__ = ["FailureSelector", "OriginalFailureSelector", "get_selector", "register_selector"]
