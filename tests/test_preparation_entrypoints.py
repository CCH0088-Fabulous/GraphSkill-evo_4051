"""Dataset entry points use the shared settings without exposing held-out data to pilots."""

import sys
from pathlib import Path

import pytest
import yaml

from repro.config import PROJECT_ROOT


@pytest.mark.parametrize("dataset", ["searchqa", "hotpotqa"])
@pytest.mark.parametrize("mode", ["parquet", "viewer"])
def test_preparation_entrypoints_use_config_and_cli_overrides(tmp_path, monkeypatch, dataset, mode):
    from scripts import prepare_hotpotqa, prepare_searchqa

    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["preparation"].update(
        raw_cache_dir=str(tmp_path / "cache"), seed=7, train_count=8, val_count=4, test_count=2
    )
    config["preparation"][f"{dataset}_revision"] = "a" * 40
    config["preparation"][f"{dataset}_download_mode"] = mode
    output = tmp_path / f"{dataset}-data"
    config["presets"][f"{dataset}.course"]["data"]["split_dir"] = str(output)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = {}

    def result(sizes):
        return {"splits": {key: {"count": value} for key, value in sizes.items()}, "source": {}}

    if dataset == "searchqa":
        def prepare(raw, manifest_dir, target, sizes, **options):
            captured.update(raw=raw, target=target, sizes=sizes, options=options)
            return result(sizes)

        monkeypatch.setattr(prepare_searchqa, "prepare_searchqa", prepare)
        main = prepare_searchqa.main
    else:
        def prepare(target, sizes, **options):
            captured.update(target=target, sizes=sizes, options=options)
            return result(sizes)

        monkeypatch.setattr(prepare_hotpotqa, "prepare_public_hotpotqa", prepare)
        main = prepare_hotpotqa.main
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(config_path), "--train", "3"])
    main()
    assert captured["target"] == output
    assert captured["sizes"] == {"train": 3, "val": 4, "test": 2}
    assert captured["options"]["seed"] == 7
    assert captured["options"]["hf_revision"] == ("a" * 40 if mode == "parquet" else None)
    assert captured["options"]["download_mode"] == mode
    assert captured["options"]["cache_dir"] == tmp_path / "cache"


def test_hotpot_course_source_uses_configured_local_paths(tmp_path, monkeypatch):
    from scripts import prepare_hotpotqa

    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["preparation"]["hotpotqa_source"] = "course"
    for split in ("train", "val", "test"):
        config["preparation"][f"hotpotqa_raw_{split}"] = str(tmp_path / f"{split}.json")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = {}

    def prepare(paths, output, sizes, **kwargs):
        captured.update(paths=paths, sizes=sizes)
        return {"splits": {key: {"count": value} for key, value in sizes.items()}, "source": {}}

    monkeypatch.setattr(prepare_hotpotqa, "prepare_hotpotqa", prepare)
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(path)])
    prepare_hotpotqa.main()
    assert captured["paths"] == {split: tmp_path / f"{split}.json" for split in ("train", "val", "test")}


def test_pilot_preparation_reads_only_training_and_validation(tmp_path, monkeypatch):
    from scripts import make_pilot_data

    reads = []
    generated = []
    settings = {"pilot_train_count": 3, "pilot_val_count": 2, "pilot_seed": 17}

    def config(_path, *, presets):
        dataset, profile = presets[-1].split(".")
        return {
            "data": {"split_dir": str(tmp_path / profile / dataset), "manifest": None},
            "preparation": settings,
        }

    def manifest(path):
        return {"dataset": Path(path).name, "source": {}}

    def read_split(_parent, split, dataset):
        reads.append((dataset, split))
        return []

    def freeze(dataset, by_split, output, sizes, **kwargs):
        generated.append((dataset, set(by_split), sizes, kwargs["seed"]))

    monkeypatch.setattr(make_pilot_data, "load_config", config)
    monkeypatch.setattr(make_pilot_data, "resolve_path", Path)
    monkeypatch.setattr(make_pilot_data, "verify_manifest", manifest)
    monkeypatch.setattr(make_pilot_data, "file_sha256", lambda path: "parent-hash")
    monkeypatch.setattr(make_pilot_data, "load_split", read_split)
    monkeypatch.setattr(make_pilot_data, "freeze_splits", freeze)
    monkeypatch.setattr(sys, "argv", ["make_pilot_data"])
    make_pilot_data.main()
    assert reads == [(name, split) for name in ("searchqa", "hotpotqa") for split in ("train", "val")]
    assert all(
        parts == {"train", "val"} and sizes["test"] == 0 and seed == 17
        for _, parts, sizes, seed in generated
    )
