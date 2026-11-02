"""Professor-guided sampling uses difficulty, fixed IDs, and immutable local receipts."""

import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from repro.datasets.hotpotqa_sampling import LEVELS, prepare_stratified_hotpotqa, select_stratified_ids
from repro.datasets.io import load_split
from repro.datasets.manifests import file_sha256, verify_manifest


def _index():
    return {f"{level}-{number}": level for level in LEVELS for number in range(6)}


def _select(training=None, development=None, **overrides):
    options = {"train_per_level": 2, "val_per_level": 1, "test_count": 3, "seed": 42, **overrides}
    return select_stratified_ids(
        _index() if training is None else training,
        [f"dev-{number}" for number in range(8)] if development is None else development,
        **options,
    )


def test_stratification_is_disjoint_and_independent_of_source_order():
    ids = _select()
    assert {split: len(rows) for split, rows in ids.items()} == {"train": 6, "val": 3, "test": 3}
    assert Counter(_index()[identity] for identity in ids["train"]) == dict.fromkeys(LEVELS, 2)
    assert Counter(_index()[identity] for identity in ids["val"]) == dict.fromkeys(LEVELS, 1)
    flattened = [identity for rows in ids.values() for identity in rows]
    assert len(set(flattened)) == len(flattened)
    assert ids == _select(dict(reversed(list(_index().items()))), [f"dev-{n}" for n in reversed(range(8))])
    assert ids != _select(seed=127)


def test_expanding_test_preserves_train_val_and_extends_existing_test_ids():
    """Synthetic ID pools isolate split selection from any model or research outputs."""
    training = {f"{level}-{number}": level for level in LEVELS for number in range(200)}
    development = [f"dev-{number}" for number in range(2000)]
    settings = {"train_per_level": 100, "val_per_level": 50, "seed": 42}
    previous = select_stratified_ids(training, development, test_count=200, **settings)
    expanded = select_stratified_ids(training, development, test_count=1000, **settings)
    assert previous["train"] == expanded["train"]
    assert previous["val"] == expanded["val"]
    assert len(expanded["test"]) == 1000
    assert previous["test"] == expanded["test"][:200]
    assert len(set(expanded["test"])) == 1000
    assert not set(expanded["test"]).intersection(expanded["train"] + expanded["val"])


@pytest.mark.parametrize(
    "training,development,message",
    [
        ({"easy-0": "easy"}, None, "Insufficient"),
        ({"bad": "unknown"}, None, "difficulty"),
        (None, ["dev-0", "dev-0"], "unique"),
        (None, ["easy-0", "dev-0", "dev-1"], "disjoint"),
        (None, ["dev-0"], "Insufficient"),
    ],
)
def test_invalid_source_pools_fail_without_substitution(training, development, message):
    with pytest.raises(ValueError, match=message):
        _select(training, development)


def _record(identity, level):
    return {
        "id": identity, "level": level, "question": "Question?", "answer": "reference-only",
        "context": {"title": ["First", "Second"], "sentences": [["first paragraph"], ["second paragraph"]]},
        "supporting_facts": {"title": ["First"], "sent_id": [0]},
    }


@pytest.fixture
def cached_source(tmp_path):
    revision = "a" * 40
    cache = tmp_path / "raw"
    base = cache / "hotpotqa--hotpot_qa" / revision / "distractor"
    base.mkdir(parents=True)
    for split, records in (
        ("train", [_record(identity, level) for identity, level in _index().items()]),
        ("validation", [_record(f"dev-{n}", "hard") for n in range(8)]),
    ):
        path = base / f"{split}-00000-of-00001.parquet"
        parquet.write_table(pa.Table.from_pylist(records), path)
        relative = f"distractor/{path.name}"
        receipt = {
            "url": f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{revision}/{relative}?download=true",
            "bytes": path.stat().st_size, "sha256": file_sha256(path),
        }
        path.with_suffix(path.suffix + ".download.json").write_text(json.dumps(receipt), encoding="utf-8")
    protocol = tmp_path / "professor.md"
    protocol.write_text("Professor sampling guidance", encoding="utf-8")
    return {
        "cache_dir": cache, "hf_revision": revision, "protocol_path": protocol,
        "train_per_level": 2, "val_per_level": 1, "test_count": 3, "seed": 42,
    }


def test_materialization_preserves_context_and_original_ids_without_overwriting(tmp_path, cached_source):
    output = tmp_path / "prepared"
    ids_path = tmp_path / "submitted/ids.json"
    manifest = prepare_stratified_hotpotqa(output, ids_output=ids_path, **cached_source)
    assert verify_manifest(output) == manifest
    assert manifest["source"]["protocol_status"] == "provisional"
    assert manifest["source"]["test_count_resolution"] == "3_user_specified"
    assert manifest["source"]["sampling"]["test_count"] == 3
    assert manifest["source"]["instruction"]["sha256"] == file_sha256(cached_source["protocol_path"])
    submitted = json.loads(ids_path.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        examples = load_split(output, split, dataset="hotpotqa")
        assert submitted[f"{split}_ids"] == [row.id for row in examples]
        assert all("first paragraph" in row.context and "second paragraph" in row.context for row in examples)
        assert all("reference-only" not in row.context for row in examples)
    original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert prepare_stratified_hotpotqa(output, ids_output=ids_path, **cached_source) == manifest
    assert original == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="different sampling protocol"):
        prepare_stratified_hotpotqa(output, ids_output=ids_path, **{**cached_source, "seed": 127})


def test_corrupted_raw_cache_is_rejected_before_any_prepared_output(tmp_path, cached_source):
    source = next(cached_source["cache_dir"].rglob("train-*.parquet"))
    source.write_bytes(source.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="receipt"):
        prepare_stratified_hotpotqa(tmp_path / "prepared", **cached_source)
    assert not (tmp_path / "prepared").exists()


def test_missing_official_shard_is_not_silently_sampled(tmp_path, cached_source):
    source = next(cached_source["cache_dir"].rglob("train-*.parquet"))
    source.rename(source.with_name("train-00000-of-00002.parquet"))
    with pytest.raises(FileNotFoundError, match="shard set is incomplete"):
        prepare_stratified_hotpotqa(tmp_path / "prepared", **cached_source)


@pytest.mark.parametrize("test_count", [200, 1000])
def test_cli_uses_provisional_preset_and_archives_exact_manifest_bytes(tmp_path, monkeypatch, test_count):
    from scripts import prepare_hotpotqa as cli

    output = tmp_path / "prepared"
    archive = tmp_path / "submitted/hotpotqa.json"
    settings = {
        "raw_cache_dir": str(tmp_path / "raw"), "hotpotqa_source": "official-public",
        "hotpotqa_download_mode": "parquet", "hotpotqa_revision": "a" * 40,
        "hotpotqa_raw_train": None, "hotpotqa_raw_val": None, "hotpotqa_raw_test": None,
        "train_count": 100, "val_count": 30, "test_count": 50, "seed": 42,
        "hotpotqa_train_per_level": 100, "hotpotqa_val_per_level": 50, "hotpotqa_test_count": test_count,
        "hotpotqa_protocol_file": str(tmp_path / "professor.md"),
        "hotpotqa_ids_output": str(tmp_path / "submitted/ids.json"),
    }
    manifest = {"source": {"protocol_status": "provisional"}, "splits": {
        split: {"count": count} for split, count in (("train", 300), ("val", 150), ("test", test_count))
    }}
    encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    def load(path, presets):
        assert presets == ["course", "hotpotqa.provisional"]
        return {"preparation": settings, "data": {"split_dir": str(output), "manifest": str(archive)}}

    def prepare(path, **options):
        assert path == output
        assert options["seed"] == 42
        assert options["train_per_level"] == 100 and options["val_per_level"] == 50
        assert options["test_count"] == test_count
        output.mkdir()
        (output / "split_manifest.json").write_bytes(encoded)
        return manifest

    monkeypatch.setattr(cli, "load_config", load)
    monkeypatch.setattr(cli, "prepare_stratified_hotpotqa", prepare)
    monkeypatch.setattr("sys.argv", ["prepare_hotpotqa.py", "--stratified"])
    cli.main()
    assert archive.read_bytes() == encoded


@pytest.mark.parametrize("bad_options", [["--overwrite"], ["--train", "100"], ["--source", "course"]])
def test_stratified_cli_rejects_incompatible_legacy_modes(monkeypatch, bad_options):
    from scripts import prepare_hotpotqa as cli

    config = {
        "data": {"split_dir": "unused"},
        "preparation": {
            "raw_cache_dir": "unused", "hotpotqa_source": "official-public",
            "hotpotqa_download_mode": "parquet", "hotpotqa_revision": "a" * 40,
            "hotpotqa_raw_train": None, "hotpotqa_raw_val": None, "hotpotqa_raw_test": None,
            "train_count": 100, "val_count": 30, "test_count": 50, "seed": 42,
        },
    }
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr("sys.argv", ["prepare_hotpotqa.py", "--stratified", *bad_options])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
