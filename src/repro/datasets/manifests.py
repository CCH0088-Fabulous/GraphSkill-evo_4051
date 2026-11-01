"""Freeze reproducible subset IDs while preserving supplied partition boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from repro.contracts import QAExample
from repro.datasets.io import SPLITS, write_examples

COURSE_LIMITS = {"train": 100, "val": 30, "test": 50}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_partitions(by_split: Mapping[str, Sequence[QAExample]], *, dataset: str) -> None:
    unknown = set(by_split).difference(SPLITS)
    if unknown:
        raise ValueError(f"Unknown partitions: {sorted(unknown)}")
    seen: dict[str, str] = {}
    for split, examples in by_split.items():
        for example in examples:
            if example.dataset != dataset or example.split != split:
                raise ValueError(f"Record {example.id!r} has inconsistent dataset or split")
            if example.id in seen:
                raise ValueError(
                    f"Duplicate or overlapping ID {example.id!r}: {seen[example.id]} and {split}"
                )
            seen[example.id] = split


def select_fixed_ids(examples: Sequence[QAExample], count: int, *, seed: int = 42) -> list[QAExample]:
    """Hash ranking is independent of source ordering and Python random versions."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Subset size must be a non-negative integer")
    if count > len(examples):
        raise ValueError(f"Requested {count} examples, but the source partition contains {len(examples)}")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Cannot select a subset from duplicate IDs")

    def rank(example: QAExample) -> tuple[str, str]:
        key = f"{seed}\0{example.dataset}\0{example.split}\0{example.id}".encode()
        return hashlib.sha256(key).hexdigest(), example.id

    return sorted(examples, key=rank)[:count]


def freeze_splits(
    dataset: str,
    by_split: Mapping[str, Sequence[QAExample]],
    output_dir: str | Path,
    sizes: Mapping[str, int],
    *,
    seed: int = 42,
    course_limits: bool = True,
    source: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write normalized JSONL and an immutable-by-default manifest of fixed IDs."""
    validate_partitions(by_split, dataset=dataset)
    if set(sizes).difference(SPLITS):
        raise ValueError("Subset sizes accept only train, val, and test keys")
    selected: dict[str, list[QAExample]] = {}
    for split in SPLITS:
        count = sizes.get(split, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid {split} subset size: {count!r}")
        if course_limits and count > COURSE_LIMITS[split]:
            raise ValueError(f"Course profile: {split} must be <= {COURSE_LIMITS[split]}")
        selected[split] = select_fixed_ids(by_split.get(split, []), count, seed=seed)
    root = Path(output_dir)
    manifest_path = root / "split_manifest.json"
    output_paths = [root / f"{split}.jsonl" for split in SPLITS]
    if not overwrite and (manifest_path.exists() or any(path.exists() for path in output_paths)):
        raise FileExistsError(
            f"{root}: frozen data already exists; use a new directory or explicit overwrite"
        )
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_examples(root / f"{split}.jsonl", selected[split])
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": seed,
        "selection": "sha256(seed\\0dataset\\0split\\0id), ascending",
        "course_profile": course_limits,
        "source": dict(source or {}),
        "synthetic": any(
            example.metadata.get("synthetic", False) for rows in selected.values() for example in rows
        ),
        "splits": {
            split: {
                "count": len(selected[split]),
                "ids": [example.id for example in selected[split]],
                "file": f"{split}.jsonl",
                "sha256": file_sha256(root / f"{split}.jsonl"),
            }
            for split in SPLITS
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(split_dir: str | Path) -> dict[str, Any]:
    """Verify file bytes, exact ID order, schema, and split isolation."""
    from repro.datasets.io import load_split

    root = Path(split_dir)
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") not in {"searchqa", "hotpotqa"}:
        raise ValueError("Unsupported split manifest schema or dataset")
    examples_by_split = {}
    for split in SPLITS:
        info = manifest["splits"][split]
        if info["file"] != f"{split}.jsonl":
            raise ValueError(f"Unexpected manifest path for {split}")
        path = root / info["file"]
        if file_sha256(path) != info["sha256"]:
            raise ValueError(f"{path}: checksum mismatch")
        examples = load_split(root, split, dataset=manifest["dataset"])
        if [example.id for example in examples] != info["ids"] or len(examples) != info["count"]:
            raise ValueError(f"{path}: manifest ID or count mismatch")
        examples_by_split[split] = examples
    validate_partitions(examples_by_split, dataset=manifest["dataset"])
    return manifest


def verify_manifest_reference(manifest: dict[str, Any], reference_path: str | Path) -> None:
    """Compare fixed data identities while allowing a relocated source checkout."""
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    actual = deepcopy(manifest)
    reference_source = reference.get("source", {})
    actual_source = actual.get("source", {})
    if isinstance(reference_source, dict) and isinstance(actual_source, dict):
        expected_ids = reference_source.get("official_manifest_sha256")
        actual_ids = actual_source.get("official_manifest_sha256")
        if expected_ids != actual_ids and isinstance(expected_ids, dict) and isinstance(actual_ids, dict):
            source_dir = Path(actual_source.get("official_manifest_dir", ""))
            if set(expected_ids) != set(actual_ids):
                raise ValueError("Official ID manifest partitions differ from the reference")
            for split, expected_digest in expected_ids.items():
                if split not in SPLITS:
                    raise ValueError("Invalid official ID manifest partition")
                content = (source_dir / split / "items.json").read_bytes()
                if hashlib.sha256(content).hexdigest() != actual_ids[split]:
                    raise ValueError("Official ID manifest does not match its recorded checksum")
                lf = content.replace(b"\r\n", b"\n")
                variants = (lf, lf.replace(b"\n", b"\r\n"))
                if expected_digest not in {hashlib.sha256(value).hexdigest() for value in variants}:
                    raise ValueError("Official ID manifest differs from the reference")
            actual_source["official_manifest_sha256"] = expected_ids
    for item in (reference, actual):
        source = item.get("source")
        if isinstance(source, dict):
            source.pop("official_manifest_dir", None)
    if reference != actual:
        raise ValueError("Dataset manifest differs from the recorded IDs, partitions, or source checksums")
