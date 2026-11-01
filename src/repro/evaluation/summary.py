"""Recompute descriptive scores from saved per-example predictions."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    source = root / "predictions.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"No predictions: {source}")
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = tuple(str(row[name]) for name in ("dataset", "split", "condition", "skill_id"))
            groups[key].append(row)
    summaries = []
    for (dataset, split, condition, skill_id), rows in sorted(groups.items()):
        latencies = [row["latency_s"] for row in rows]
        summaries.append(
            {
                "dataset": dataset,
                "split": split,
                "condition": condition,
                "skill_id": skill_id,
                "n_predictions": len(rows),
                "n_unique_examples": len({row["sample_id"] for row in rows}),
                "em": statistics.mean(row["em"] for row in rows),
                "f1": statistics.mean(row["f1"] for row in rows),
                "latency_mean_s": statistics.mean(latencies),
                "latency_median_s": statistics.median(latencies),
                "failed_predictions": sum(row["status"] == "error" for row in rows),
                "truncated_predictions": sum(bool(row.get("output_truncated")) for row in rows),
            }
        )
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    return {
        "run": root.name,
        "synthetic": config["experiment"]["profile"] == "synthetic",
        "groups": summaries,
        "note": (
            "Evolution groups may contain repeated training evaluations; "
            "use selected-skill evaluation for final results."
        ),
    }
