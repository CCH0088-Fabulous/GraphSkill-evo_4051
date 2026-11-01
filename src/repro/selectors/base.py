"""Failure-batch selection contract, separate from evolutionary parent selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class FailureSelector(Protocol):
    def select(self, rows: Sequence[dict[str, Any]], *, k: int, seed: int) -> list[dict[str, Any]]:
        """Return at most k training failures without rewriting optimizer inputs."""
        ...
