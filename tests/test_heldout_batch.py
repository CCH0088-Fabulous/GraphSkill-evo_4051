"""Frozen held-out evaluation accepts completed evidence and never invokes optimization."""

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from repro.artifacts.fingerprints import file_sha256
from repro.config import load_config
from repro.contracts import QAExample
from repro.datasets.manifests import freeze_splits
from repro.experiments import heldout


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _development(tmp_path, monkeypatch, dataset="hotpotqa"):
    batch = tmp_path / "development"
    batch.mkdir()
    runtime = tmp_path / "runtime.py"
    runtime.write_bytes(b"original runtime\n")
    source = {"project_python_sha256": {"runtime.py": file_sha256(runtime)}}
    _json(batch / "source.json", source)
    data = tmp_path / "data"
    sizes = heldout._DATASET_SIZES[dataset]
    examples = {
        split: [QAExample(f"{split}-{i}", dataset, split, "q", "c", ("a",)) for i in range(count)]
        for split, count in sizes.items()
    }
    manifest = freeze_splits(dataset, examples, data, sizes, course_limits=False)
    reference = tmp_path / "reference.json"
    reference.write_bytes((data / "split_manifest.json").read_bytes())
    identity = {"fingerprint_sha256": "original-model", "precision": "bfloat16",
                "attention_implementation": "sdpa", "device": "cuda:0"}
    monkeypatch.setattr(heldout, "capture_model_identity", lambda *args, **kwargs: deepcopy(identity))
    preset = "hotpotqa.provisional" if dataset == "hotpotqa" else "searchqa.course"
    base = load_config("config.yaml", presets=["course", preset])
    base["data"].update(split_dir=str(data), manifest=str(reference))
    base["experiment"].update(output_root=str(tmp_path / "outputs"))
    plans, results = {}, []
    for seed in heldout.SEEDS:
        cfg = deepcopy(base)
        cfg["experiment"]["seed"] = seed
        plans[str(seed)] = {dataset: cfg}
        for condition in (*heldout.CONDITIONS, "selected_val"):
            run = batch / "runs" / f"seed-{seed}" / dataset / condition / "run"
            actual = deepcopy(cfg)
            actual["experiment"].update(
                condition="original" if condition == "selected_val" else condition,
                output_root=str(run.parent),
            )
            saved = {"status": "completed", "n": sizes["val"], "split": "val", "em": 0.7, "f1": 0.8}
            if condition == "original":
                saved["best"] = {"id": "g0001_i0000", "val_hard": 0.7, "val_soft": 0.8}
            for name, content in (("config.json", actual), ("run_summary.json", saved),
                                  ("model_identity.json", identity), ("data_manifest.json", manifest),
                                  ("source.json", source)):
                _json(run / name, content)
            if condition == "initial_skill":
                skill = run / "skills/evaluated.md"
                skill.parent.mkdir(parents=True)
                skill.write_bytes(b"initial skill\n")
                (run / "skills.jsonl").write_text(json.dumps({"skill_id": "evaluated",
                    "sha256": file_sha256(skill)}) + "\n", encoding="utf-8")
            if condition == "original":
                skill = run / "upstream/best_skill.md"
                skill.parent.mkdir()
                skill.write_bytes(f"selected skill {seed}\r\n".encode())
                _json(batch / f"selection/seed-{seed}/{dataset}.json", {
                    "path": str(skill), "sha256": file_sha256(skill), "selection_split": "val",
                    "test_examples_used": False,
                })
            if condition == "selected_val":
                skill = run / "skills/evaluated.md"
                skill.parent.mkdir(parents=True)
                skill.write_bytes(f"selected skill {seed}\n".encode())
            results.append({"seed": seed, "condition": condition, "path": str(run),
                            "status": "completed", "summary": saved})
    _json(batch / "summary.json", {"status": "completed", "error": None,
                                   "experiment_seeds": list(heldout.SEEDS), "runs": results})
    _json(batch / "scope.json", {"split": "val", "test_examples_sent_to_model": False})
    _json(batch / "plan.json", {"seeds": list(heldout.SEEDS), "configs_by_seed": plans})
    return batch


@pytest.mark.parametrize("dataset", ["hotpotqa", "searchqa"])
def test_preflight_freezes_all_seeds_without_exposing_test_to_optimization(tmp_path, monkeypatch, dataset):
    batch = _development(tmp_path, monkeypatch, dataset)
    plan = heldout.load_heldout_plan(batch)
    assert tuple(map(int, plan["configs"])) == heldout.SEEDS
    assert plan["data_manifest"]["splits"]["test"]["count"] == heldout._DATASET_SIZES[dataset]["test"]
    assert not any(cfg["experiment"]["allow_test"] for cfg in plan["configs"].values())
    assert plan["selected"]["0"]["content"] == b"selected skill 0\r\n"


@pytest.mark.parametrize(
    "change", ["incomplete", "subset", "already_tested", "skill", "initial", "data", "model"]
)
@pytest.mark.parametrize("dataset", ["hotpotqa", "searchqa"])
def test_preflight_rejects_changed_or_incomplete_evidence(tmp_path, monkeypatch, change, dataset):
    batch = _development(tmp_path, monkeypatch, dataset)
    summary = json.loads((batch / "summary.json").read_text())
    if change == "incomplete":
        summary["status"] = "incomplete"
        _json(batch / "summary.json", summary)
    elif change == "subset":
        summary["experiment_seeds"] = [42]
        _json(batch / "summary.json", summary)
    elif change == "already_tested":
        _json(batch / "scope.json", {"split": "test", "test_examples_sent_to_model": True})
    elif change == "skill":
        skill = batch / f"runs/seed-0/{dataset}/original/run/upstream/best_skill.md"
        skill.write_bytes(b"reselected after test")
    elif change == "initial":
        (batch / f"runs/seed-0/{dataset}/initial_skill/run/skills/evaluated.md").write_bytes(b"new initial")
    elif change == "data":
        with (tmp_path / "data/test.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif change == "model":
        monkeypatch.setattr(heldout, "capture_model_identity", lambda *args, **kwargs: {
            "fingerprint_sha256": "different-model"})
    with pytest.raises(ValueError):
        heldout.load_heldout_plan(batch)


def test_code_edits_after_development_do_not_block_preflight(tmp_path, monkeypatch):
    batch = _development(tmp_path, monkeypatch)
    (tmp_path / "runtime.py").write_bytes(b"edited after development\n")
    assert heldout.load_heldout_plan(batch)["data_manifest"]["splits"]["test"]["count"] == 1000


@pytest.mark.parametrize("state", ["completed", "completed_with_errors", "failed", "wrong_model"])
@pytest.mark.parametrize("dataset", ["hotpotqa", "searchqa"])
def test_run_uses_one_server_test_only_frozen_skills_and_stops_after_failure(
    tmp_path, monkeypatch, state, dataset,
):
    batch = _development(tmp_path, monkeypatch, dataset)
    monkeypatch.setattr(heldout, "environment_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(heldout, "source_snapshot", lambda: {})
    server_calls, calls = [], []

    @contextmanager
    def server(settings, directory):
        server_calls.append("started")
        yield ["fixture-server"]
        server_calls.append("stopped")

    def evaluate(config, *, split, saved_skill):
        assert split == "test" and config["experiment"]["allow_test"]
        assert "frozen/data" in config["data"]["split_dir"].replace("\\", "/")
        assert Path(config["data"]["manifest"]).is_file()
        condition, seed = config["experiment"]["condition"], config["experiment"]["seed"]
        calls.append((seed, condition))
        if condition == "no_skill":
            assert saved_skill is None
        else:
            assert saved_skill.is_file() and "frozen/skills" in saved_skill.as_posix()
            if condition == "initial_skill":
                assert config["experiment"]["initial_skill"] == str(saved_skill)
        if state == "failed":
            raise RuntimeError("fixture failure")
        run = Path(config["experiment"]["output_root"]) / "result"
        _json(run / "run_summary.json", {"status": state if state == "completed_with_errors" else "completed",
                                         "n": heldout._DATASET_SIZES[dataset]["test"], "split": "test"})
        identity = {"fingerprint_sha256": "wrong" if state == "wrong_model" else "original-model"}
        _json(run / "model_identity.json", identity)
        _json(run / "data_manifest.json", json.loads(Path(config["data"]["manifest"]).read_text()))
        return run

    monkeypatch.setattr(heldout, "managed_transformers_server", server)
    monkeypatch.setattr(heldout, "run_baseline", evaluate)
    report = heldout.run_heldout(batch)
    summary = json.loads((report / "summary.json").read_text())
    assert server_calls == ["started", "stopped"]
    failed = state in {"failed", "wrong_model"}
    assert len(calls) == (1 if failed else 9)
    assert summary["status"] == ("incomplete" if failed else state)
    assert summary["optimization_allowed"] is False
    assert summary["test_feedback_to_method_development_allowed"] is False
    assert not summary["member_b_improved_and_ablation_included"]
    assert len(summary["runs"]) == 9
    assert summary["dataset"] == dataset
    assert summary["n_per_run"] == heldout._DATASET_SIZES[dataset]["test"]
    if failed:
        assert all(row["status"] == "not_run" for row in summary["runs"][1:])
    assert json.loads((batch / "summary.json").read_text())["status"] == "completed"


def _fixture_runtime(monkeypatch, calls, *, fail_at=None):
    @contextmanager
    def server(settings, directory):
        yield ["fixture-server"]

    def evaluate(config, *, split, saved_skill):
        key = (config["experiment"]["seed"], config["experiment"]["condition"])
        calls.append(key)
        if key == fail_at:
            raise RuntimeError("interrupted")
        run = Path(config["experiment"]["output_root"]) / f"run-{len(calls)}"
        manifest = json.loads(Path(config["data"]["manifest"]).read_text())
        _json(run / "run_summary.json", {"status": "completed",
                                         "n": manifest["splits"]["test"]["count"], "split": "test"})
        _json(run / "model_identity.json", {"fingerprint_sha256": "original-model"})
        _json(run / "data_manifest.json", json.loads(Path(config["data"]["manifest"]).read_text()))
        return run

    monkeypatch.setattr(heldout, "managed_transformers_server", server)
    monkeypatch.setattr(heldout, "run_baseline", evaluate)


@pytest.mark.parametrize("dataset", ["hotpotqa", "searchqa"])
def test_resume_keeps_finished_subruns_and_reruns_only_the_rest(tmp_path, monkeypatch, dataset):
    batch = _development(tmp_path, monkeypatch, dataset)
    monkeypatch.setattr(heldout, "environment_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(heldout, "source_snapshot", lambda: {})
    first_calls: list = []
    _fixture_runtime(monkeypatch, first_calls, fail_at=(0, "original"))
    report = heldout.run_heldout(batch)
    assert len(first_calls) == 3
    assert json.loads((report / "summary.json").read_text())["status"] == "incomplete"
    partial_runs = sorted(p.name for p in (report / "runs").rglob("run-*"))

    resumed_calls: list = []
    _fixture_runtime(monkeypatch, resumed_calls)
    again = heldout.resume_heldout(report)
    assert again == report
    assert len(resumed_calls) == 7 and (0, "no_skill") not in resumed_calls
    summary = json.loads((report / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["resume_attempt"] == 1
    assert len(summary["runs"]) == 9
    assert (report / "summary.attempt-1.json").is_file()
    assert set(partial_runs) <= {p.name for p in (report / "runs").rglob("run-*")}
    assert (report / "server_command-resume-1.json").is_file()


def test_resume_rejects_changed_frozen_inputs(tmp_path, monkeypatch):
    batch = _development(tmp_path, monkeypatch)
    monkeypatch.setattr(heldout, "environment_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(heldout, "source_snapshot", lambda: {})
    _fixture_runtime(monkeypatch, [], fail_at=(0, "initial_skill"))
    report = heldout.run_heldout(batch)
    (report / "frozen/skills/seed-0.md").write_bytes(b"reselected after seeing test")
    with pytest.raises(ValueError):
        heldout.resume_heldout(report)


@pytest.mark.parametrize("arguments", [["--seeds", "42"], ["--set", "model.temperature=0"],
                                       ["--config", "other.yaml"], ["--provisional-hotpotqa"],
                                       ["--dataset", "searchqa"]])
def test_test_cli_rejects_reconfiguration_or_seed_selection(arguments):
    from scripts.run_formal import main

    with pytest.raises(SystemExit) as error:
        main(["--test-from-batch", "source", *arguments])
    assert error.value.code == 2
