"""Candidate ancestry, separate from execution routing inside each skill."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from repro.visualization.skill_graph import _label


def lineage_to_mermaid(records: Iterable[dict[str, Any]]) -> str:
    individuals: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = str(record.get("skill_id", record.get("id", "")))
        if not identifier:
            raise ValueError("Lineage record needs skill_id or id")
        if identifier in individuals:
            raise ValueError(f"Duplicate lineage individual: {identifier}")
        individuals[identifier] = record
    identifiers = dict.fromkeys(individuals)
    for record in individuals.values():
        for parent in record.get("parent_ids", []):
            identifiers.setdefault(str(parent), None)
    names = {identifier: f"s{index}" for index, identifier in enumerate(identifiers)}
    lines = ["flowchart LR"]
    for identifier, node_name in names.items():
        record = individuals.get(identifier, {})
        label = identifier
        for field in ("generation", "operator", "origin", "val_hard", "retained"):
            if field in record:
                label += f"\n{field}: {record[field]}"
        safe_label = _label(label)
        lines.append(f'    {node_name}["{safe_label}"]')
    for identifier, record in individuals.items():
        for parent in record.get("parent_ids", []):
            lines.append(f"    {names[str(parent)]} --> {names[identifier]}")
    return "\n".join(lines) + "\n"
