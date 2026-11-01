"""Evaluate one frozen skill or the no-skill baseline without optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repro.artifacts.gpu import GPUMonitor
from repro.config import resolve_path
from repro.datasets.io import load_split
from repro.experiments.context import (
    finish_run,
    prepare_run,
    require_allowed_split,
    skill_path,
    verify_run_service,
)


def run_baseline(
    config: dict[str, Any],
    *,
    split: str = "val",
    saved_skill: Path | None = None,
    offline: bool = False,
) -> Path:
    from repro.integration.qa_adapter import execute_examples

    require_allowed_split(config, split)
    condition = config["experiment"]["condition"]
    if saved_skill is not None and condition == "no_skill":
        raise ValueError("no_skill cannot be evaluated with a skill file")
    if saved_skill is None and condition not in {"no_skill", "initial_skill"}:
        raise ValueError("Saved skill is required to evaluate an evolved condition")
    content = (
        "" if condition == "no_skill" else (saved_skill or skill_path(config)).read_text(encoding="utf-8")
    )
    if saved_skill is not None and condition == "initial_skill":
        if content != skill_path(config).read_text(encoding="utf-8"):
            raise ValueError("initial_skill must match the configured initial skill, not an evolved artifact")
    labels = {"synthetic": config["experiment"]["profile"] == "synthetic", "backend_offline": offline}
    store, client = prepare_run(config, offline=offline)
    try:
        verify_run_service(store, client, offline=offline)
        record = store.save_skill("evaluated", content, condition=condition)
        examples = load_split(resolve_path(config["data"]["split_dir"]), split, config["data"]["dataset"])
        if not examples:
            raise ValueError(f"No examples in {split}")
        with GPUMonitor(lambda row: store.append_jsonl("gpu.jsonl", row)):
            rows = execute_examples(
                examples,
                content,
                client,
                store,
                condition,
                skill_id=record["sha256"],
                allow_test=config["experiment"]["allow_test"],
            )
        if all(row["status"] == "error" for row in rows):
            raise RuntimeError("All prediction requests failed; this run does not provide valid QA results")
        result = {
            "condition": condition,
            "dataset": config["data"]["dataset"],
            "split": split,
            "n": len(rows),
            "em": sum(row["hard"] for row in rows) / len(rows),
            "f1": sum(row["soft"] for row in rows) / len(rows),
            **labels,
        }
    except Exception as exc:
        finish_run(store, client, labels, error=exc)
        raise
    finish_run(store, client, result)
    return store.path
