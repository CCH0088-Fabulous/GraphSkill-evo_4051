"""Verify sampling-instruction content while allowing its local filename to change."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from repro.datasets.manifests import file_sha256


def _has_instruction(source: dict[str, Any]) -> bool:
    instruction = source.get("instruction")
    return (
        isinstance(instruction, dict)
        and isinstance(instruction.get("path"), str)
        and bool(instruction["path"].strip())
        and isinstance(instruction.get("sha256"), str)
        and bool(instruction["sha256"])
    )


def same_source_after_instruction_rename(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
    """Ignore only the instruction's location; compare every other provenance field exactly."""
    if not _has_instruction(recorded) or not _has_instruction(current):
        return False
    relocated = deepcopy(recorded)
    relocated["instruction"]["path"] = current["instruction"]["path"]
    return relocated == current


def verify_instruction(instruction: dict[str, Any], current_path: Path) -> None:
    """The configured file must retain the recorded exact bytes, regardless of filename."""
    if (
        not _has_instruction({"instruction": instruction})
        or instruction["sha256"] != file_sha256(current_path)
    ):
        raise ValueError("HotpotQA teacher instruction differs from the recorded protocol")
