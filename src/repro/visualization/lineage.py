"""Candidate ancestry, separate from execution routing inside each skill."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from repro.visualization.skill_graph import _label


def run_lineage_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read one evolution run's saved history into lineage records; scores are validation-only."""
    run = Path(run_dir)
    rows = [
        json.loads(line)
        for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("History is empty")
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    best = summary.get("best", {}).get("id")
    final = rows[-1].get("next_population") or rows[-1].get("population") or []
    retained = {item["id"] for item in final}
    individuals: dict[str, dict[str, Any]] = {}
    for row in rows:
        for item in (*row.get("population", []), *row.get("offspring", [])):
            if item["id"] in individuals:
                continue
            score = item.get("val_hard")
            individuals[item["id"]] = {
                "skill_id": item["id"], "generation": item.get("generation"),
                "parent_ids": list(item.get("parent_ids", [])), "operator": item.get("origin"),
                "val_hard": None if score is None else round(score, 4),
                "retained": item["id"] in retained, "selected": item["id"] == best,
            }
    return list(individuals.values())


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
        for field in ("generation", "operator", "origin", "val_hard", "retained", "selected"):
            if field in record:
                label += f"\n{field}: {record[field]}"
        safe_label = _label(label)
        lines.append(f'    {node_name}["{safe_label}"]')
    for identifier, record in individuals.items():
        for parent in dict.fromkeys(str(item) for item in record.get("parent_ids", [])):
            lines.append(f"    {names[parent]} --> {names[identifier]}")
    return "\n".join(lines) + "\n"
