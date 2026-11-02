"""Prepare one isolated run with frozen data, environment, and source snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro.artifacts.model_identity import capture_model_identity
from repro.artifacts.provenance import environment_snapshot, source_snapshot
from repro.artifacts.store import RunStore
from repro.config import PROJECT_ROOT, resolve_path
from repro.datasets.manifests import COURSE_LIMITS, verify_manifest, verify_manifest_reference
from repro.inference.accounting import UsageLedger
from repro.inference.client import ChatClient
from repro.inference.offline import OfflineClient


def prepare_run(config: dict[str, Any], *, offline: bool = False) -> tuple[RunStore, Any]:
    data_root = resolve_path(config["data"]["split_dir"])
    manifest = verify_manifest(data_root)
    if manifest["dataset"] != config["data"]["dataset"]:
        raise ValueError("Configured dataset does not match the frozen manifest")
    if config["data"].get("manifest"):
        verify_manifest_reference(manifest, resolve_path(config["data"]["manifest"]))
    is_synthetic = bool(manifest.get("synthetic"))
    if offline and not is_synthetic:
        raise ValueError("Offline fixture backend can only be used with synthetic data")
    if is_synthetic != (config["experiment"]["profile"] == "synthetic"):
        raise ValueError("Synthetic datasets require experiment.profile=synthetic and vice versa")
    if config["experiment"]["profile"] == "course":
        for split, limit in COURSE_LIMITS.items():
            if manifest["splits"][split]["count"] > limit:
                raise ValueError(f"Course profile: {split} contains more than {limit} examples")
    identity = capture_model_identity(config["model"], synthetic=is_synthetic)
    store = RunStore(resolve_path(config["experiment"]["output_root"]))
    store.save_snapshot(config, environment_snapshot(offline=offline), source_snapshot())
    store.save_json("data_manifest.json", manifest)
    store.save_json("model_identity.json", identity)
    store.record_event("data_verified", dataset=manifest["dataset"], synthetic=is_synthetic)
    budget = config["budget"]
    ledger = UsageLedger(
        max_calls=budget["max_calls"],
        max_completion_tokens=budget["max_completion_tokens"],
        sink=lambda row: store.append_jsonl("calls.jsonl", row),
    )
    client = (
        OfflineClient(ledger)
        if offline
        else ChatClient(
            config["model"],
            ledger,
            seed=config["experiment"]["seed"],
        )
    )
    return store, client


def require_allowed_split(config: dict[str, Any], split: str) -> None:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if split == "test" and not config["experiment"]["allow_test"]:
        raise ValueError(
            "Held-out test is disabled. Freeze methods/settings before setting experiment.allow_test=true"
        )


def verify_run_service(store: RunStore, client: Any, *, offline: bool) -> None:
    if offline:
        store.save_json("service.json", {"synthetic": True, "model_loaded": False})
    else:
        store.save_json("service.json", client.verify_service())


def finish_run(
    store: RunStore, client: Any, result: dict[str, Any], *, error: Exception | None = None
) -> None:
    prediction_file = store.path / "predictions.jsonl"
    predictions = (
        [
            json.loads(line)
            for line in prediction_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if prediction_file.is_file()
        else []
    )
    failures = sum(row["status"] == "error" for row in predictions)
    summary = {
        **result,
        "usage": client.ledger.summary(),
        "run_id": store.run_id,
        "status": "failed" if error else ("completed_with_errors" if failures else "completed"),
        "failed_predictions": failures,
        "truncated_predictions": sum(bool(row.get("output_truncated")) for row in predictions),
        "saved_predictions": len(predictions),
        "error": str(error) if error else None,
    }
    store.save_json("run_summary.json", summary)
    store.record_event("run_failed" if error else "run_completed", error=str(error) if error else None)
    client.close()


def skill_path(config: dict[str, Any]) -> Path:
    explicit = config["experiment"]["initial_skill"]
    if explicit:
        return resolve_path(explicit)
    if config["data"]["dataset"] == "searchqa":
        return PROJECT_ROOT / "third_party/GraphSkillEvo/graphskillevo/initial_skills/searchqa/init_skill.md"
    return PROJECT_ROOT / "skills/hotpotqa/initial.md"
