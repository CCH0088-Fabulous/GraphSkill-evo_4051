"""Create an immutable protocol version only after a two-round acceptance passes."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from repro.artifacts.acceptance import audit_acceptance
from repro.artifacts.fingerprints import file_sha256
from repro.artifacts.model_identity import capture_model_identity
from repro.artifacts.provenance import environment_snapshot, source_snapshot
from repro.artifacts.serialization import json_text, utc_now
from repro.config import PROJECT_ROOT, load_config, resolve_path
from repro.datasets.manifests import verify_manifest
from repro.freezing.environment import installed_versions


def publish_freeze(version: str, acceptance_run: Path, *, seeds: list[int] | None = None) -> Path:
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("Freeze version must be v1, v2, ...")
    destination = PROJECT_ROOT / "manifests" / f"experiment-{version}"
    if destination.exists():
        raise FileExistsError(f"Frozen versions cannot be overwritten: {destination}")
    seeds = seeds or [42]
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Frozen seeds must be nonnegative integers")
    audit = audit_acceptance(acceptance_run)
    if audit["status"] != "passed":
        raise ValueError(f"Acceptance failed: {audit['issues']}")
    for relative, digest in audit["accepted_runtime_sources_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise ValueError(f"Runtime code changed after acceptance: {relative}")
    shared = load_config(PROJECT_ROOT / "config.yaml")
    if shared["model"] != audit["model_settings"]:
        raise ValueError("Shared model settings differ from the successfully accepted settings")
    identity = capture_model_identity(shared["model"], synthetic=False)
    if identity["fingerprint_sha256"] != audit["model_fingerprint"]:
        raise ValueError("Model files changed after acceptance")
    reference = (destination / "manifest.json").relative_to(PROJECT_ROOT).as_posix()
    snapshots: dict[str, dict[str, Any]] = {}
    datasets = {}
    for dataset in ("searchqa", "hotpotqa"):
        config = load_config(PROJECT_ROOT / f"configs/{dataset}.course.yaml")
        if config["experiment"]["freeze_manifest"] != reference:
            raise ValueError("Course configurations must reference the version being published")
        if config["model"] != shared["model"] or config["experiment"]["allow_test"]:
            raise ValueError("Formal configurations must share the accepted model and keep test disabled")
        directory = resolve_path(config["data"]["split_dir"])
        datasets[dataset] = verify_manifest(directory)
        for condition in ("no_skill", "initial_skill", "original"):
            snapshots[f"{dataset}.{condition}"] = load_config(
                PROJECT_ROOT / f"configs/{dataset}.course.yaml",
                [
                    f"experiment.condition={condition}",
                    f"experiment.seed={seeds[0]}",
                ],
            )
    protected = set((PROJECT_ROOT / "src/repro").rglob("*.py"))
    protected.update(p for p in (PROJECT_ROOT / "scripts").iterdir() if p.suffix in {".py", ".ps1", ".sh"})
    for name in (
        "config.yaml",
        "configs/searchqa.course.yaml",
        "configs/hotpotqa.course.yaml",
        "requirements/lock.windows.txt",
        "third_party/upstream.lock.json",
        "third_party/hotpotqa.lock.json",
        "skills/hotpotqa/initial.md",
        "pyproject.toml",
        "third_party/GraphSkillEvo/graphskillevo/initial_skills/searchqa/init_skill.md",
        "data/processed/searchqa/split_manifest.json",
        "data/processed/hotpotqa/split_manifest.json",
        f"docs/frozen-settings-{version}.md",
    ):
        protected.add(PROJECT_ROOT / name)
    hashes = {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in sorted(protected)}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "version": version,
        "created_at": utc_now(),
        "scope": "Shared protocol and member A conditions; B's methods are pending implementation",
        "seeds": seeds,
        "default_seed": seeds[0],
        "configs": {},
        "reserved_conditions": {
            "improved": "not implemented or approved",
            "ablation": "not implemented or approved",
        },
        "rules": {
            "target_truncation": "flag raw output; empty scored answer; EM/F1 zero; no continuation",
            "optimizer_truncation": "reject and retry within the existing phase attempt budget",
            "optimizer_seed": (
                "first try uses base seed; retries use the first 32 bits of "
                "SHA256([optimizer_retry_v1,base,phase,stage,stage_attempt])"
            ),
            "target_seed": "fixed experiment seed for each QA request",
            "http_retry": "same payload and seed; transport errors, HTTP 429/5xx only; every attempt metered",
            "budget_exhaustion": "fail and stop; preserve partial artifacts; no selector substitution",
            "test": "disabled by default; explicit final evaluation after method and skill selection freeze",
            "runtime_overrides": "only output_root, an approved seed, and explicit allow_test may differ",
        },
        "acceptance": audit,
        "model_identity": identity,
        "datasets": datasets,
        "environment": {**environment_snapshot(offline=False), "installed_packages": installed_versions()},
        "source": source_snapshot(),
        "protected_files_sha256": hashes,
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "configs").mkdir()
    for name, config in snapshots.items():
        target = destination / "configs" / f"{name}.json"
        target.write_text(json_text(config, pretty=True) + "\n", encoding="utf-8")
        manifest["configs"][name] = {"path": f"configs/{name}.json", "sha256": file_sha256(target)}
    path = destination / "manifest.json"
    path.write_text(json_text(manifest, pretty=True) + "\n", encoding="utf-8")
    path.with_suffix(".sha256").write_text(file_sha256(path) + "\n", encoding="utf-8")
    activation = {
        "version": version,
        "manifest": reference,
        "sha256": file_sha256(path),
        "datasets": {name: snapshots[f"{name}.original"]["data"]["split_dir"] for name in datasets},
    }
    active = PROJECT_ROOT / "manifests/active.json"
    if active.exists():
        (destination / "previous-active.json").write_bytes(active.read_bytes())
    temporary = destination / "activation.json.tmp"
    temporary.write_text(json_text(activation, pretty=True) + "\n", encoding="utf-8")
    os.replace(temporary, active)
    return path
