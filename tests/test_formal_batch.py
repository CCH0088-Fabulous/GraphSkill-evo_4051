"""Formal comparisons snapshot settings, preserve failures, and never infer on test."""

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from repro.config import load_config
from repro.experiments import formal


@pytest.mark.parametrize("failure", [False, True])
def test_formal_batch_uses_snapshot_and_skips_selection_after_failure(tmp_path, monkeypatch, failure):
    configs = {
        name: load_config("config.yaml", presets=["course", f"{name}.course"])
        for name in formal.DATASETS
    }
    for config in configs.values():
        config["experiment"]["output_root"] = str(tmp_path)
    originals = deepcopy(configs)
    calls = []
    monkeypatch.setattr(formal, "load_formal_configs", lambda path, overrides: configs)
    monkeypatch.setattr(formal, "environment_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(formal, "source_snapshot", lambda: {})

    @contextmanager
    def server(settings, directory):
        yield ["fixture-server"]

    def execute(config, *, split="val", saved_skill=None):
        assert split == "val" and not config["experiment"]["allow_test"]
        name = config["data"]["dataset"]
        condition = config["experiment"]["condition"]
        calls.append((name, condition, saved_skill is not None))
        assert config["evolution"]["generations"] == 3
        if failure and name == "searchqa" and condition == "original" and saved_skill is None:
            raise RuntimeError("fixture evolution failure")
        run = Path(config["experiment"]["output_root"]) / "run"
        run.mkdir(parents=True)
        (run / "run_summary.json").write_text(json.dumps({"status": "completed", "usage": {"calls": 0}}))
        if condition == "original" and saved_skill is None:
            (run / "upstream").mkdir()
            (run / "upstream/best_skill.md").write_text("fixture selected skill")
        config["evolution"]["generations"] = 99
        return run

    monkeypatch.setattr(formal, "managed_transformers_server", server)
    monkeypatch.setattr(formal, "run_baseline", execute)
    monkeypatch.setattr(formal, "run_evolution_experiment", execute)
    report = formal.run_formal()
    result = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == ("incomplete" if failure else "completed")
    assert result["test_examples_sent_to_model"] is False
    assert result["temporary_server_stopped"] is True
    assert len(result["runs"]) == 8
    assert ("hotpotqa", "original", True) in calls
    assert (("searchqa", "original", True) in calls) is not failure
    assert configs == originals
    assert json.loads((report / "config.json").read_text(encoding="utf-8")) == originals


def test_formal_preflight_rejects_test_enabled_before_starting_server(monkeypatch):
    config = load_config("config.yaml", presets=["course", "searchqa.course"])
    config["experiment"]["allow_test"] = True
    monkeypatch.setattr(formal, "load_config", lambda *args, **kwargs: deepcopy(config))
    with pytest.raises(ValueError, match="test disabled"):
        formal.load_formal_configs("config.yaml")
