"""Pilot batch boundaries: both datasets, isolated outputs and no held-out examples."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from repro.experiments.pilot import run_pilots


@pytest.mark.parametrize("fail_searchqa", [False, True])
def test_pilot_runs_both_datasets_and_evaluates_only_completed_skills(tmp_path, monkeypatch, fail_searchqa):
    from repro.experiments import pilot

    invocations = []

    def config(path, overrides=()):
        dataset = Path(path).name.split(".")[0]
        result = {
            "data": {"dataset": dataset, "split_dir": dataset},
            "model": {"backend": "transformers", "temperature": 0.0},
            "evolution": {"generations": 1},
            "experiment": {"allow_test": False, "condition": "original"},
        }
        for override in overrides:
            field, value = override.split("=", 1)
            section, key = field.split(".")
            result[section][key] = yaml.safe_load(value)
        return result

    def manifest(path):
        return {"dataset": str(path), "synthetic": False, "splits": {"test": {"count": 0}}}

    @contextmanager
    def server(_settings, _directory):
        assert _settings["temperature"] == 0.6
        yield ["fake-server"]

    def write_run(cfg, skill=False):
        root = Path(cfg["experiment"]["output_root"]) / "run"
        root.mkdir(parents=True)
        (root / "run_summary.json").write_text(json.dumps({"status": "completed", "usage": {"calls": 0}}))
        if skill:
            (root / "upstream").mkdir()
            (root / "upstream/best_skill.md").write_text("fake skill")
        return root

    def baseline(cfg, *, split, saved_skill=None):
        assert split == "val" and not cfg["experiment"]["allow_test"]
        assert cfg["model"]["temperature"] == 0.6
        invocations.append((cfg["data"]["dataset"], cfg["experiment"]["condition"], bool(saved_skill)))
        return write_run(cfg)

    def evolution(cfg):
        assert cfg["model"]["temperature"] == 0.6
        dataset = cfg["data"]["dataset"]
        invocations.append((dataset, "original", False))
        if dataset == "searchqa" and fail_searchqa:
            raise RuntimeError("fixture failure")
        return write_run(cfg, skill=True)

    monkeypatch.setattr(pilot, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pilot, "load_config", config)
    monkeypatch.setattr(pilot, "verify_manifest", manifest)
    monkeypatch.setattr(pilot, "resolve_path", lambda path: path)
    monkeypatch.setattr(pilot, "managed_transformers_server", server)
    monkeypatch.setattr(pilot, "run_baseline", baseline)
    monkeypatch.setattr(pilot, "run_evolution_experiment", evolution)
    result = json.loads((run_pilots(["model.temperature=0.6"]) / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == ("incomplete" if fail_searchqa else "completed")
    assert result["test_examples_used"] is False
    assert len(result["runs"]) == 8
    assert ("hotpotqa", "original", True) in invocations
    assert (("searchqa", "original", True) in invocations) is not fail_searchqa
    if fail_searchqa:
        assert [row["status"] for row in result["runs"] if row["dataset"] == "searchqa"] == [
            "completed",
            "completed",
            "failed",
            "skipped",
        ]


@pytest.mark.parametrize("status,expected", [("completed", 0), ("incomplete", 1)])
def test_pilot_cli_reports_incomplete_batch_as_failure(tmp_path, monkeypatch, status, expected):
    from scripts import run_pilots as entrypoint

    (tmp_path / "summary.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    monkeypatch.setattr(entrypoint, "run_pilots", lambda overrides, profile: tmp_path)
    assert entrypoint.main([]) == expected
