"""Data leakage, provenance, integrity, and source adaptation checks."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from repro.contracts import QAExample
from repro.datasets import downloads, hub
from repro.datasets.hotpotqa import format_context, prepare_hotpotqa, prepare_public_hotpotqa
from repro.datasets.io import (
    example_from_record,
    load_examples,
    load_split,
    select_parquet_rows,
    write_examples,
)
from repro.datasets.manifests import (
    freeze_splits,
    select_fixed_ids,
    validate_partitions,
    verify_manifest,
    verify_manifest_reference,
)
from repro.datasets.searchqa import prepare_searchqa


def example(identity: str, split: str = "train", dataset: str = "searchqa") -> QAExample:
    return QAExample(identity, dataset, split, "A question?", "Some evidence.", ("answer",))


def write_json(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_subset_selection_is_order_independent_and_partition_aware() -> None:
    items = [example(str(index)) for index in range(20)]
    selected = select_fixed_ids(items, 5, seed=7)
    assert selected == select_fixed_ids(list(reversed(items)), 5, seed=7)
    assert selected != select_fixed_ids(items, 5, seed=8)
    assert [item.id for item in items] == [str(index) for index in range(20)]


@pytest.mark.parametrize("count", [-1, True, 1.5, 3])
def test_invalid_subset_sizes_fail(count: int) -> None:
    with pytest.raises(ValueError):
        select_fixed_ids([example("a"), example("b")], count)


def test_partition_overlap_rejected_before_sampling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlapping"):
        freeze_splits(
            "searchqa",
            {"train": [example("same")], "val": [example("same", "val")]},
            tmp_path,
            {"train": 0, "val": 0},
        )
    assert not (tmp_path / "split_manifest.json").exists()


def test_manifest_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    freeze_splits("searchqa", {"train": [example("a")]}, tmp_path, {"train": 1})
    manifest = verify_manifest(tmp_path)
    assert manifest["splits"]["train"]["ids"] == ["a"]
    with pytest.raises(FileExistsError):
        freeze_splits("searchqa", {"train": [example("b")]}, tmp_path, {"train": 1})
    assert load_split(tmp_path, "train")[0].id == "a"
    (tmp_path / "train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_manifest(tmp_path)


def test_recorded_data_identity_allows_source_relocation_but_rejects_drift(tmp_path: Path) -> None:
    source = {
        "official_manifest_dir": "E:/old/checkout",
        "revision": "revision-1",
        "raw_files": {"train.parquet": {"sha256": "original-hash"}},
    }
    actual = freeze_splits(
        "searchqa", {"train": [example("a")]}, tmp_path / "data", {"train": 1}, source=source
    )
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(actual), encoding="utf-8")
    actual["source"]["official_manifest_dir"] = "D:/new/checkout"
    verify_manifest_reference(actual, reference)
    actual["source"]["raw_files"]["train.parquet"]["sha256"] = "changed-hash"
    with pytest.raises(ValueError, match="source checksums"):
        verify_manifest_reference(actual, reference)
    actual["source"]["raw_files"]["train.parquet"]["sha256"] = "original-hash"
    actual["splits"]["train"]["ids"] = ["different-id"]
    with pytest.raises(ValueError, match="partitions"):
        verify_manifest_reference(actual, reference)


def test_official_id_manifest_accepts_only_equivalent_line_endings(tmp_path: Path) -> None:
    source_dir = tmp_path / "upstream"
    identity_file = source_dir / "train" / "items.json"
    identity_file.parent.mkdir(parents=True)
    lf = b'[\n  {"id": "a"}\n]\n'
    identity_file.write_bytes(lf)
    source = {
        "official_manifest_dir": str(source_dir),
        "official_manifest_sha256": {"train": hashlib.sha256(lf).hexdigest()},
    }
    actual = freeze_splits(
        "searchqa", {"train": [example("a")]}, tmp_path / "data", {"train": 1}, source=source
    )
    reference = deepcopy(actual)
    reference["source"]["official_manifest_sha256"]["train"] = hashlib.sha256(
        lf.replace(b"\n", b"\r\n")
    ).hexdigest()
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    verify_manifest_reference(actual, path)
    changed = lf.replace(b'"a"', b'"b"')
    identity_file.write_bytes(changed)
    actual["source"]["official_manifest_sha256"]["train"] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="differs from the reference"):
        verify_manifest_reference(actual, path)


def test_course_profile_is_explicitly_optional(tmp_path: Path) -> None:
    items = [example(str(index)) for index in range(101)]
    with pytest.raises(ValueError, match="Course profile"):
        freeze_splits("searchqa", {"train": items}, tmp_path, {"train": 101})
    manifest = freeze_splits(
        "searchqa",
        {"train": items},
        tmp_path,
        {"train": 101},
        course_limits=False,
    )
    assert manifest["course_profile"] is False


@pytest.mark.parametrize("field,value", [("id", ""), ("context", []), ("answers", []), ("split", "dev")])
def test_normalized_records_are_strict(field: str, value: object) -> None:
    record = example("a").to_dict()
    record[field] = value
    with pytest.raises(ValueError):
        example_from_record(record)


def test_io_roundtrip_detects_duplicates_and_wrong_partition(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_examples(path, [example("a")])
    assert load_examples(path) == [example("a")]
    with pytest.raises(ValueError, match="not val"):
        write_examples(tmp_path / "val.jsonl", [example("b")])
        load_split(tmp_path, "val")
    write_examples(path, [example("a"), example("a")])
    with pytest.raises(ValueError, match="duplicate"):
        load_examples(path)


def test_searchqa_restores_manifest_boundaries_not_raw_source_split(tmp_path: Path) -> None:
    manifests = tmp_path / "ids"
    for split, identity in (("train", "a"), ("val", "b"), ("test", "c")):
        write_json(manifests / split / "items.json", [{"id": identity}])
    raw = write_json(
        tmp_path / "raw.json",
        [
            {
                "key": identity,
                "question": "Question?",
                "context": "Evidence",
                "answers": ["answer"],
                "split": "upstream_train",
            }
            for identity in ("c", "b", "a")
        ],
    )
    output = tmp_path / "prepared"
    prepare_searchqa(raw, manifests, output, {"train": 1, "val": 1, "test": 1})
    assert load_split(output, "val")[0].id == "b"
    assert load_split(output, "test")[0].metadata["source_record"]["split"] == "upstream_train"
    verify_manifest(output)


def test_searchqa_missing_requested_manifest_id_fails(tmp_path: Path) -> None:
    manifests = tmp_path / "ids"
    for split, records in (("train", [{"id": "missing"}]), ("val", []), ("test", [])):
        write_json(manifests / split / "items.json", records)
    with pytest.raises(ValueError, match="missing 1 selected"):
        prepare_searchqa(write_json(tmp_path / "raw.json", []), manifests, tmp_path / "out", {"train": 1})


def hotpot_record(identity: str) -> dict:
    return {
        "_id": identity,
        "question": "Who?",
        "answer": "Ada",
        "context": [["Book", ["Ada wrote it."]]],
        "supporting_facts": [["Book", 0]],
    }


def test_hotpot_context_formats_are_equivalent() -> None:
    expected = "[Book]\nFirst. Second."
    assert format_context([["Book", ["First.", "Second."]]]) == expected
    assert format_context({"title": ["Book"], "sentences": [["First.", "Second."]]}) == expected
    with pytest.raises(ValueError, match="columns"):
        format_context({"title": ["Book"], "sentences": []})


def test_public_hotpot_is_disclosed_and_disjoint(tmp_path: Path) -> None:
    train = write_json(tmp_path / "train.json", [hotpot_record(str(index)) for index in range(8)])
    dev = write_json(tmp_path / "dev.json", [hotpot_record("dev-only")])
    output = tmp_path / "out"
    manifest = prepare_public_hotpotqa(
        output,
        {"train": 3, "val": 2, "test": 1},
        raw_train_path=train,
        raw_dev_path=dev,
    )
    assert manifest["course_split_verified"] is False
    assert manifest["source"]["heldout_source"] == "official_validation"
    assert {item.id for item in load_split(output, "train")}.isdisjoint(
        item.id for item in load_split(output, "val")
    )
    heldout = load_split(output, "test")[0]
    assert heldout.id == "dev-only"
    assert heldout.metadata["source_record"]["supporting_facts"] == [["Book", 0]]
    verify_manifest(output)


def test_course_hotpot_requires_supplied_partition_and_detects_overlap(tmp_path: Path) -> None:
    path = write_json(tmp_path / "data.json", [hotpot_record("same")])
    with pytest.raises(ValueError, match="file is required"):
        prepare_hotpotqa({"train": path}, tmp_path / "out", {"train": 1, "val": 1})
    with pytest.raises(ValueError, match="overlapping"):
        prepare_hotpotqa({"train": path, "val": path}, tmp_path / "out", {"train": 1, "val": 1})


def test_viewer_retrieval_limits_queries_and_records_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/api/datasets/"):
            return httpx.Response(200, json={"sha": "fixed-revision"})
        if request.url.path == "/splits":
            return httpx.Response(200, json={"splits": [{"config": "default", "split": "train"}]})
        if request.url.path == "/filter":
            assert request.url.params["where"] == "\"key\"='a' OR \"key\"='b'"
            return httpx.Response(
                200,
                json={
                    "rows": [{"row": {"key": identity}, "truncated_cells": []} for identity in ("a", "b")],
                    "num_rows_total": 2,
                },
            )
        raise AssertionError(request.url)

    original_client = httpx.Client
    monkeypatch.setattr(
        hub.httpx,
        "Client",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    records, source = hub.viewer_records_by_ids("example/data", ["a", "b"])
    assert [record["key"] for record in records] == ["a", "b"]
    assert source["revision_pinned_api"] is False
    assert len(requests) == 4


def test_fixture_data_is_marked_synthetic_and_has_no_heldout_results() -> None:
    for dataset in ("searchqa", "hotpotqa"):
        root = Path("tests/fixtures") / dataset
        manifest = verify_manifest(root)
        assert manifest["synthetic"] is True
        assert load_split(root, "test") == []
        assert all(item.metadata["synthetic"] for item in load_split(root, "train"))


def test_partition_dataset_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        validate_partitions({"train": [replace(example("a"), dataset="hotpotqa")]}, dataset="searchqa")


def test_download_resumes_partial_file_and_verifies_cached_bytes(tmp_path: Path) -> None:
    target = tmp_path / "source.parquet"
    target.with_suffix(".parquet.part").write_bytes(b"abc")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["range"] == "bytes=3-"
        return httpx.Response(206, content=b"def", headers={"content-range": "bytes 3-5/6"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = downloads.download_file(client, "https://example.org/source.parquet", target, 6)
        assert target.read_bytes() == b"abcdef"
        assert receipt["bytes"] == 6
        # Completed immutable cache does not require another request.
        assert downloads.download_file(client, "https://example.org/source.parquet", target, 6) == receipt
        target.write_bytes(b"abcdeg")
        with pytest.raises(ValueError, match="checksum mismatch"):
            downloads.download_file(client, "https://example.org/source.parquet", target, 6)


def test_metadata_retries_without_changing_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(429 if len(calls) == 1 else 200, json={"result": "ok"})

    monkeypatch.setattr(downloads.time, "sleep", sleeps.append)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert downloads.get_json(client, "https://example.org/rows", offset=42) == {"result": "ok"}
    assert len(calls) == 2 and calls[0] == calls[1]
    assert sleeps == [2]


def test_parquet_row_offsets_preserve_viewer_selection(tmp_path: Path) -> None:
    import random

    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    paths = []
    for shard in range(2):
        path = tmp_path / f"train-{shard:05}.parquet"
        parquet.write_table(
            arrow.Table.from_pylist([{"id": index} for index in range(shard * 5, shard * 5 + 5)]), path
        )
        paths.append(path)
    rows, metadata = select_parquet_rows(paths, 4, seed=42)
    expected = sorted(random.Random(42).sample(range(10), 4))
    assert metadata["selected_row_offsets"] == expected
    assert [row["id"] for row in rows] == expected
