"""Frozen experiments reject settings, source and model drift before inference."""

import json
from copy import deepcopy

import pytest

from repro.artifacts.fingerprints import file_sha256
from repro.config import DEFAULTS, load_config
from repro.freezing.verification import validate_frozen_config, validate_model_fingerprint


def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr("repro.freezing.verification.environment_snapshot", lambda offline: {"packages": {}})
    monkeypatch.setattr("repro.freezing.verification.installed_versions", lambda: {})
    config = deepcopy(DEFAULTS)
    config["experiment"]["freeze_manifest"] = "frozen/manifest.json"
    folder = tmp_path / "frozen"
    folder.mkdir()
    snapshot = folder / "config.json"
    snapshot.write_text(json.dumps(config), encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("# fixture source", encoding="utf-8")
    manifest = {
        "version": "v1",
        "protected_files_sha256": {"source.py": file_sha256(source)},
        "environment": {"packages": {}, "installed_packages": {}},
        "seeds": [42],
        "configs": {"searchqa.original": {"path": "config.json", "sha256": file_sha256(snapshot)}},
        "model_identity": {"fingerprint_sha256": "fixture-fingerprint"},
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.with_suffix(".sha256").write_text(file_sha256(path), encoding="utf-8")
    return config


def test_frozen_settings_allow_only_predeclared_runtime_changes(tmp_path, monkeypatch):
    config = frozen(tmp_path, monkeypatch)
    config["experiment"].update(output_root="other-output", allow_test=True)
    result = validate_frozen_config(config, root=tmp_path)
    assert result["version"] == "v1"
    validate_model_fingerprint(result, {"fingerprint_sha256": "fixture-fingerprint"})
    with pytest.raises(ValueError, match="weights"):
        validate_model_fingerprint(result, {"fingerprint_sha256": "other"})
    config["model"]["temperature"] = 0.9
    with pytest.raises(ValueError, match="model.temperature"):
        validate_frozen_config(config, root=tmp_path)


def test_frozen_sources_and_seed_list_are_checked(tmp_path, monkeypatch):
    config = frozen(tmp_path, monkeypatch)
    config["experiment"]["seed"] = 43
    with pytest.raises(ValueError, match="seed list"):
        validate_frozen_config(config, root=tmp_path)
    config["experiment"]["seed"] = 42
    (tmp_path / "source.py").write_text("# changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        validate_frozen_config(config, root=tmp_path)


def test_declared_freeze_cannot_be_disabled_from_cli(tmp_path):
    path = tmp_path / "formal.yaml"
    path.write_text("experiment:\n  freeze_manifest: frozen/manifest.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be overridden"):
        load_config(path, ["experiment.freeze_manifest=null"])


def test_declared_data_identity_cannot_be_disabled_from_cli(tmp_path):
    path = tmp_path / "formal.yaml"
    path.write_text("data:\n  manifest: recorded-data.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data manifest cannot be overridden"):
        load_config(path, ["data.manifest=null"])


def test_freeze_publication_requires_passed_acceptance_and_new_version(tmp_path, monkeypatch):
    from repro.freezing import publication

    monkeypatch.setattr(publication, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        publication, "audit_acceptance", lambda path: {"status": "failed", "issues": ["failed"]}
    )
    with pytest.raises(ValueError, match="Acceptance failed"):
        publication.publish_freeze("v1", tmp_path)
    folder = tmp_path / "manifests/experiment-v1"
    folder.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        publication.publish_freeze("v1", tmp_path)


def test_incomplete_acceptance_cannot_be_published(tmp_path):
    from repro.artifacts.acceptance import audit_acceptance

    (tmp_path / "summary.json").write_text(json.dumps({"status": "incomplete", "runs": []}), encoding="utf-8")
    result = audit_acceptance(tmp_path)
    assert result["status"] == "failed" and result["issues"]


def test_legacy_activation_does_not_enforce_freeze_when_not_requested(tmp_path):
    config = deepcopy(DEFAULTS)
    folder = tmp_path / "manifests"
    folder.mkdir()
    (folder / "active.json").write_text(
        json.dumps(
            {
                "datasets": {"searchqa": config["data"]["split_dir"]},
            }
        ),
        encoding="utf-8",
    )
    assert validate_frozen_config(config, root=tmp_path) is None


def test_frozen_dependencies_cannot_change_without_version_update(tmp_path, monkeypatch):
    config = frozen(tmp_path, monkeypatch)
    monkeypatch.setattr("repro.freezing.verification.installed_versions", lambda: {"tokenizers": "different"})
    with pytest.raises(ValueError, match="dependency versions"):
        validate_frozen_config(config, root=tmp_path)
