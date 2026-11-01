"""Artifact integrity checks for traceable and non-overwriting experiments."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from repro.artifacts import RunStore
from repro.artifacts.serialization import json_text


def test_snapshots_redact_credentials_but_keep_accounting(tmp_path):
    store = RunStore(tmp_path, "trial-1")
    store.save_snapshot(
        {
            "model": {"api_key": "private-key"}, "HF_TOKEN": "hf-secret",
            "max_completion_tokens": 100, "token_budget": 500,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
        {"authorization": "Bearer test"},
        {"commit": "abc"},
    )
    config = json.loads((store.path / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["api_key"] == "[REDACTED]"
    assert config["max_completion_tokens"] == 100
    assert config["token_budget"] == 500
    assert config["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert config["HF_TOKEN"] == "[REDACTED]"
    assert "private-key" not in (store.path / "config.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.save_json("config.json", {"different": True})
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, "trial-1")


@pytest.mark.parametrize("path", ["../outside.json", "nested/../../outside.json"])
def test_run_store_rejects_path_traversal(tmp_path, path):
    store = RunStore(tmp_path, "run")
    with pytest.raises(ValueError, match="escapes"):
        store.save_json(path, {})
    assert not (tmp_path / "outside.json").exists()


def test_jsonl_writes_are_thread_safe_and_skills_are_hashed(tmp_path):
    store = RunStore(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.append_jsonl("events.jsonl", {"i": i}), range(100)))
    rows = [json.loads(line) for line in (store.path / "events.jsonl").read_text().splitlines()]
    assert {row["i"] for row in rows} == set(range(100))
    skill = store.save_skill("g0000_i0000", "skill text", parent_ids=[], generation=0)
    assert len(skill["sha256"]) == 64
    assert (store.path / skill["path"]).read_text() == "skill text"
    with pytest.raises(FileExistsError):
        store.save_skill("g0000_i0000", "changed text")


def test_duplicate_predictions_are_rejected(tmp_path):
    store = RunStore(tmp_path)
    row = {
        "dataset": "searchqa", "split": "val", "condition": "original",
        "skill_id": "g0", "sample_id": "1", "answer": "A",
    }
    store.write_prediction(row)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        store.write_prediction({**row, "answer": "B"})
    assert len((store.path / "predictions.jsonl").read_text().splitlines()) == 1


def test_nonfinite_metrics_and_embedded_credentials_are_handled():
    with pytest.raises(ValueError):
        json_text({"em": float("nan")})
    assert "sk-abcdefghijklmnopqrst" not in json_text({"error": "key=sk-abcdefghijklmnopqrst"})
    assert "Bearer abc123" not in json_text({"error": "header: Bearer abc123"})


def test_nested_upstream_directory_does_not_require_overwriting_existing_files(tmp_path):
    store = RunStore(tmp_path)
    upstream = store.path / "upstream" / "generations" / "generation_0001"
    upstream.mkdir(parents=True)
    (upstream / "generation_record.json").write_text('{"upstream": true}', encoding="utf-8")
    relative = "upstream/generations/generation_0001/repro_metadata.json"
    store.save_json(relative, {"run_id": store.run_id})
    assert (store.path / relative).is_file()
    with pytest.raises(FileExistsError):
        store.save_json("upstream/generations/generation_0001/generation_record.json", {})
    assert json.loads((upstream / "generation_record.json").read_text()) == {"upstream": True}


def test_alternate_data_stream_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        RunStore(tmp_path).save_text("snapshot.json:hidden", "do not write")
