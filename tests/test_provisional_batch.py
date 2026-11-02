"""Provisional stratification, seed isolation and disabled held-out inference boundaries."""

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from repro.artifacts.fingerprints import file_sha256
from repro.config import load_config
from repro.contracts import QAExample
from repro.datasets.manifests import freeze_splits
from repro.experiments import formal, provisional


def _protocol(tmp_path, monkeypatch, *, test_count=1000):
    instruction = tmp_path / "teacher.md"
    instruction.write_text("Stratified sampling; user-specified test count 1000\n", encoding="utf-8")
    source = {
        "kind": "official_public_stratified",
        "config": "distractor",
        "protocol_status": "provisional",
        "course_split_verified": False,
        "test_count_resolution": "1000_user_specified",
        "instruction": {"path": str(instruction), "sha256": file_sha256(instruction)},
        "sampling": {
            "method": "sha256_seeded_id_rank_v1",
            "difficulty_counts": {
                "train": {"easy": 100, "medium": 100, "hard": 100},
                "val": {"easy": 50, "medium": 50, "hard": 50},
            },
            "test_count": 1000,
            "seed": 42,
        },
    }
    records = {}
    for split, count in (("train", 300), ("val", 150), ("test", test_count)):
        records[split] = [
            QAExample(
                f"{split}-{index}", "hotpotqa", split, "question", "context", ("answer",),
                {"level": ("easy", "medium", "hard")[index % 3]},
            )
            for index in range(count)
        ]
    data = tmp_path / "data"
    manifest = freeze_splits(
        "hotpotqa", records, data, {split: len(rows) for split, rows in records.items()},
        source=source, course_limits=False,
    )
    reference = tmp_path / "reference.json"
    reference.write_bytes((data / "split_manifest.json").read_bytes())
    config = load_config("config.yaml", presets=["course", "hotpotqa.course"])
    config["experiment"].update(profile="custom", output_root=str(tmp_path / "outputs"))
    config["data"].update(split_dir=str(data), manifest=str(reference))
    config["preparation"]["hotpotqa_protocol_file"] = str(instruction)
    monkeypatch.setattr(provisional, "load_config", lambda *args, **kwargs: deepcopy(config))
    return config, manifest


def _save_manifest(config, manifest):
    data = Path(config["data"]["split_dir"])
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (data / "split_manifest.json").write_text(encoded, encoding="utf-8")
    Path(config["data"]["manifest"]).write_text(encoded, encoding="utf-8")


def test_provisional_preflight_accepts_real_stratified_ids_and_keeps_test_disabled(tmp_path, monkeypatch):
    config, manifest = _protocol(tmp_path, monkeypatch)
    actual, source = provisional.load_provisional_config("config.yaml")
    assert actual == config
    assert source == manifest["source"]
    assert not actual["experiment"]["allow_test"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "official_public_custom_split"),
        ("protocol_status", "confirmed"),
        ("course_split_verified", True),
        ("test_count_resolution", "200_pending_teacher_confirmation"),
    ],
)
def test_provisional_preflight_rejects_unstratified_or_misrepresented_protocol(
    tmp_path, monkeypatch, field, value,
):
    config, manifest = _protocol(tmp_path, monkeypatch)
    manifest["source"][field] = value
    _save_manifest(config, manifest)
    with pytest.raises(ValueError, match="provisional stratified distractor"):
        provisional.load_provisional_config("config.yaml")


def test_provisional_preflight_rejects_wrong_partition_counts(tmp_path, monkeypatch):
    _protocol(tmp_path, monkeypatch, test_count=200)
    with pytest.raises(ValueError, match="300/150/1000"):
        provisional.load_provisional_config("config.yaml")


def test_provisional_preflight_checks_actual_difficulty_labels(tmp_path, monkeypatch):
    config, manifest = _protocol(tmp_path, monkeypatch)
    path = Path(config["data"]["split_dir"]) / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["metadata"]["level"] = "unknown"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest["splits"]["train"]["sha256"] = file_sha256(path)
    _save_manifest(config, manifest)
    with pytest.raises(ValueError, match="difficulty counts"):
        provisional.load_provisional_config("config.yaml")


def test_provisional_preflight_rejects_changed_teacher_instruction(tmp_path, monkeypatch):
    config, _ = _protocol(tmp_path, monkeypatch)
    Path(config["preparation"]["hotpotqa_protocol_file"]).write_text("revised sizes", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher instruction differs"):
        provisional.load_provisional_config("config.yaml")


def test_provisional_test_flag_is_rejected_before_serving(tmp_path, monkeypatch):
    config, _ = _protocol(tmp_path, monkeypatch)
    config["experiment"]["allow_test"] = True
    with pytest.raises(ValueError, match="test disabled"):
        provisional.run_provisional("config.yaml", seeds=[0])
    assert not Path(config["experiment"]["output_root"]).exists()


@pytest.mark.parametrize("seeds", [[], [0, 0], [-1], [False], [0, "42"]])
def test_invalid_seed_plan_fails_before_creating_outputs(tmp_path, seeds):
    config = load_config("config.yaml", presets=["course", "hotpotqa.course"])
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    with pytest.raises(ValueError, match="distinct nonnegative integers"):
        formal.run_comparison_batch({"hotpotqa": config}, seeds=seeds)
    assert not (tmp_path / "outputs").exists()


def test_multiseed_plan_uses_one_service_isolates_skills_and_preserves_failed_seed(tmp_path, monkeypatch):
    config = load_config("config.yaml", presets=["course", "hotpotqa.course"])
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    original = deepcopy(config)
    calls, servers = [], []
    monkeypatch.setattr(formal, "environment_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(formal, "source_snapshot", lambda: {})

    @contextmanager
    def server(settings, directory):
        servers.append(directory)
        yield ["fixture-server"]

    def execute(config, *, split="val", saved_skill=None):
        assert split == "val" and not config["experiment"]["allow_test"]
        seed, condition = config["experiment"]["seed"], config["experiment"]["condition"]
        calls.append((seed, condition, saved_skill))
        if saved_skill is not None:
            assert f"seed-{seed}" in str(saved_skill)
        run = Path(config["experiment"]["output_root"]) / "fixture-run"
        run.mkdir(parents=True)
        (run / "run_summary.json").write_text(json.dumps({"status": "completed", "usage": {"calls": 0}}))
        if condition == "original" and saved_skill is None:
            if seed == 0:
                raise RuntimeError("fixture seed zero evolution failed")
            (run / "upstream").mkdir()
            (run / "upstream/best_skill.md").write_text(f"skill for seed {seed}")
        config["evolution"]["generations"] = 99
        return run

    monkeypatch.setattr(formal, "managed_transformers_server", server)
    monkeypatch.setattr(formal, "run_baseline", execute)
    monkeypatch.setattr(formal, "run_evolution_experiment", execute)
    report = formal.run_comparison_batch({"hotpotqa": config}, seeds=[0, 42, 127])
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    plan = json.loads((report / "plan.json").read_text(encoding="utf-8"))
    assert len(servers) == 1 and config == original
    assert summary["status"] == "incomplete" and summary["temporary_server_stopped"]
    assert len(summary["runs"]) == 12
    assert [row["status"] for row in summary["runs"] if row["seed"] == 0] == [
        "completed", "completed", "failed", "skipped",
    ]
    assert all(row["status"] == "completed" for row in summary["runs"] if row["seed"] in (42, 127))
    assert len(calls) == 11
    for seed in (0, 42, 127):
        assert plan["configs_by_seed"][str(seed)]["hotpotqa"]["experiment"]["seed"] == seed
        assert plan["configs_by_seed"][str(seed)]["hotpotqa"]["evolution"]["generations"] == 3


@pytest.mark.parametrize("arguments", [["--seeds", "0"], [
    "--provisional-hotpotqa", "--seeds", "0", "--set", "experiment.seed=42",
]])
def test_cli_rejects_ambiguous_seed_or_protocol_selection(arguments):
    from scripts import run_formal

    with pytest.raises(SystemExit) as error:
        run_formal.main(arguments)
    assert error.value.code == 2
