"""Check two-round pilot evidence before publishing shared experiment settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repro.artifacts.fingerprints import file_sha256
from repro.inference.seeding import optimizer_retry_seed


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_acceptance(batch: Path) -> dict[str, Any]:
    batch = batch.resolve()
    summary = _json(batch / "summary.json")
    issues: list[str] = []
    findings: list[dict[str, Any]] = []
    evidence: dict[str, str] = {}
    models: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    runtime_sources: list[dict[str, str]] = []
    expected = {
        (d, c)
        for d in ("searchqa", "hotpotqa")
        for c in ("no_skill", "initial_skill", "original", "selected_val")
    }
    actual = {(r["dataset"], r["condition"]) for r in summary["runs"]}
    if actual != expected or len(summary["runs"]) != 8 or summary["status"] != "completed":
        issues.append("All eight acceptance runs must complete")
    if not summary.get("temporary_server_stopped"):
        issues.append("Temporary service cleanup was not confirmed")
    if issues:
        return {"status": "failed", "issues": issues, "findings": [], "batch": str(batch),
                "scope": "Incomplete batch; detailed partial artifacts are preserved"}
    totals = {k: 0 for k in ("calls", "prompt_tokens", "completion_tokens")}
    val_ids: dict[str, list[set[str]]] = {"searchqa": [], "hotpotqa": []}
    for run in summary["runs"]:
        if not run.get("path"):
            issues.append(f"Missing run: {run['dataset']}/{run['condition']}")
            continue
        path = Path(run["path"]).resolve()
        if not path.is_relative_to(batch):
            issues.append("Run evidence escapes the acceptance batch")
            continue
        result, config = _json(path / "run_summary.json"), _json(path / "config.json")
        model = _json(path / "model_identity.json")
        source = _json(path / "source.json")["project_python_sha256"]
        runtime_sources.append(
            {
                k: v
                for k, v in source.items()
                if k.startswith(
                    (
                        "src/repro/inference/",
                        "src/repro/serving/",
                        "src/repro/integration/",
                        "src/repro/evaluation/",
                        "src/repro/selectors/",
                        "src/repro/experiments/",
                    )
                )
            }
        )
        models.append(config["model"])
        fingerprints.add(model.get("fingerprint_sha256", ""))
        usage = result["usage"]
        if result["status"] != "completed" or not usage["usage_complete"] or usage["failed_calls"]:
            issues.append(f"Incomplete run or model accounting: {path.name}")
        if config["experiment"]["allow_test"]:
            issues.append("Acceptance must keep test disabled")
        for key in totals:
            totals[key] += usage[key]
        predictions = _rows(path / "predictions.jsonl")
        manifest = _json(path / "data_manifest.json")
        if manifest.get("synthetic") or manifest["splits"]["test"]["count"]:
            issues.append("Acceptance requires real pilot data with no test examples")
        allowed_ids = {split: set(spec["ids"]) for split, spec in manifest["splits"].items()}
        for row in predictions:
            if row["split"] not in {"train", "val"} or row["sample_id"] not in allowed_ids[row["split"]]:
                issues.append("Prediction uses an example outside the frozen train/val partitions")
            if row.get("output_truncated"):
                if row["em"] != 0 or row["f1"] != 0 or row.get("failure_type") != "output_truncated":
                    issues.append("Truncated answer was not an explicit zero-score failure")
                findings.append(
                    {
                        "type": "target_truncated",
                        "dataset": run["dataset"],
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                    }
                )
        if run["condition"] != "original":
            val_ids[run["dataset"]].append({r["sample_id"] for r in predictions})
        calls = _rows(path / "calls.jsonl")
        finished = [r for r in calls if r["event"] == "call_finished"]
        if len(finished) != usage["calls"] or len(calls) != 2 * usage["calls"]:
            issues.append("Call ledger is missing starts or finishes")
        for row in finished:
            meta = row.get("metadata", {})
            if meta.get("split") == "test":
                issues.append("Model request received a test example")
            if row["role"] == "optimizer":
                expected_seed = optimizer_retry_seed(
                    config["experiment"]["seed"],
                    meta.get("phase", ""),
                    meta.get("stage", ""),
                    meta.get("seed_attempt", 0),
                )
                if row.get("request", {}).get("seed") != expected_seed:
                    issues.append("Optimizer request seed differs from the frozen derivation rule")
        if run["condition"] == "original":
            history = _json(path / "upstream/history.json")
            if len(history) != 2:
                issues.append("Acceptance must finish two evolution rounds")
            operators = [r["operator"] for h in history for r in h["operator_records"]]
            if operators != ["mutation_other", "mutation_graph", "crossover_other", "crossover_graph"]:
                issues.append("Acceptance did not exercise all four operators in order")
            if any(len(h["offspring"]) != 2 for h in history):
                issues.append("Each acceptance round must evaluate two new candidates")
            for op in sorted((path / "operators").glob("call_*.json")):
                record = _json(op)
                if record.get("seed_policy") != "optimizer_retry_v1":
                    issues.append("Optimizer did not use the approved retry seed policy")
                if record.get("output_truncated"):
                    findings.append({"type": "optimizer_truncated_rejected", "path": str(op)})
            for name in ("upstream/history.json", "upstream/best_skill.md", "graphs/lineage.mmd"):
                evidence[str(path / name)] = file_sha256(path / name)
        for name in (
            "config.json",
            "source.json",
            "model_identity.json",
            "data_manifest.json",
            "run_summary.json",
            "calls.jsonl",
            "predictions.jsonl",
        ):
            evidence[str(path / name)] = file_sha256(path / name)
    if not models or any(m != models[0] for m in models):
        issues.append("Acceptance conditions use different model settings")
    if len(fingerprints) != 1 or "" in fingerprints:
        issues.append("Acceptance did not pin one exact model identity")
    if not runtime_sources or not runtime_sources[0] or any(s != runtime_sources[0] for s in runtime_sources):
        issues.append("Acceptance runs use different or unrecorded runtime sources")
    for dataset, groups in val_ids.items():
        if len(groups) != 3 or any(group != groups[0] for group in groups):
            issues.append(f"Validation IDs differ between {dataset} conditions")
    evidence[str(batch / "summary.json")] = file_sha256(batch / "summary.json")
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "findings": findings,
        "batch": str(batch),
        "model_settings": models[0] if models else None,
        "model_fingerprint": next(iter(fingerprints)) if len(fingerprints) == 1 else None,
        "totals": totals,
        "evidence_sha256": evidence,
        "test_examples_used": False,
        "accepted_runtime_sources_sha256": runtime_sources[0] if runtime_sources else {},
        "scope": "Two-round train/validation acceptance; not research results",
    }
