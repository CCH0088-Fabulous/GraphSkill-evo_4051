"""Reject drift from a published experiment protocol before model requests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from repro.artifacts.fingerprints import file_sha256
from repro.artifacts.provenance import environment_snapshot
from repro.config import PROJECT_ROOT, resolve_path
from repro.freezing.environment import installed_versions


def read_manifest(path: Path, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = path.resolve()
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if file_sha256(path) != expected:
        raise ValueError("Frozen manifest checksum differs; publish a new version")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, digest in manifest["protected_files_sha256"].items():
        target = resolve_path(relative, root=root)
        if not target.is_relative_to(root.resolve()) or not target.is_file() or file_sha256(target) != digest:
            raise ValueError(f"Frozen experiment file changed: {relative}; publish a new version")
    installed = environment_snapshot(offline=False)["packages"]
    if installed != manifest["environment"]["packages"]:
        raise ValueError("Installed package versions differ from the frozen experiment")
    if installed_versions() != manifest["environment"]["installed_packages"]:
        raise ValueError("Installed dependency versions differ from the frozen experiment")
    return manifest


def validate_frozen_config(config: dict[str, Any], *, root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    reference = config["experiment"].get("freeze_manifest")
    if not reference:
        return None
    activation_path = root / "manifests/active.json"
    activation = (
        json.loads(activation_path.read_text(encoding="utf-8")) if activation_path.is_file() else None
    )
    path = resolve_path(reference, root=root)
    if activation and resolve_path(activation["manifest"], root=root) == path:
        if file_sha256(path) != activation["sha256"]:
            raise ValueError("Active frozen manifest checksum changed")
    manifest = read_manifest(path, root=root)
    dataset, condition = config["data"]["dataset"], config["experiment"]["condition"]
    key = f"{dataset}.{condition}"
    if key not in manifest["configs"]:
        raise ValueError(
            f"Condition {key} is not approved in this freeze; B's pending methods need a new version"
        )
    item = manifest["configs"][key]
    snapshot = path.parent / item["path"]
    if not snapshot.resolve().is_relative_to(path.parent) or file_sha256(snapshot) != item["sha256"]:
        raise ValueError("Frozen configuration snapshot changed")
    expected = json.loads(snapshot.read_text(encoding="utf-8"))
    actual = deepcopy(config)
    if actual["experiment"]["seed"] not in manifest["seeds"]:
        raise ValueError("Run seed is not in the frozen seed list")
    for name in ("seed", "output_root", "allow_test"):
        expected["experiment"][name] = actual["experiment"][name]
    if actual != expected:
        differences = [
            f"{section}.{key}"
            for section in expected
            for key in expected[section]
            if expected[section][key] != actual.get(section, {}).get(key)
        ]
        raise ValueError(f"Run settings differ from frozen {manifest['version']}: {', '.join(differences)}")
    return {
        "version": manifest["version"],
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "expected_model_fingerprint": manifest["model_identity"]["fingerprint_sha256"],
    }


def validate_model_fingerprint(frozen: dict[str, Any] | None, identity: dict[str, Any]) -> None:
    if frozen and identity.get("fingerprint_sha256") != frozen["expected_model_fingerprint"]:
        raise ValueError("Local weights, tokenizer or model configuration differ from the frozen model")
