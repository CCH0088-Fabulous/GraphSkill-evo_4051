"""Build disjoint, difficulty-stratified HotpotQA splits from a pinned local cache."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from repro.contracts import QAExample
from repro.datasets.hotpotqa import hotpotqa_example
from repro.datasets.io import SPLITS, iter_records
from repro.datasets.manifests import file_sha256, freeze_splits, verify_manifest
from repro.datasets.protocol import same_source_after_instruction_rename

LEVELS = ("easy", "medium", "hard")
SAMPLER = "sha256_seeded_id_rank_v1"


def select_stratified_ids(
    training: Mapping[str, str],
    development: Sequence[str],
    *,
    train_per_level: int,
    val_per_level: int,
    test_count: int,
    seed: int,
) -> dict[str, list[str]]:
    """Select from IDs and difficulty alone, independently of input order or answers."""
    for value in (train_per_level, val_per_level, test_count, seed):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Sampling sizes and seed must be non-negative integers")
    if len(set(development)) != len(development) or set(training).intersection(development):
        raise ValueError("Official source IDs must be unique and train/dev disjoint")
    if set(training.values()).difference(LEVELS):
        raise ValueError("Training records must use original easy, medium, or hard difficulty labels")

    def ranked(ids: Sequence[str], role: str, level: str = "") -> list[str]:
        def rank(identity: str) -> tuple[bytes, str]:
            key = json.dumps([SAMPLER, seed, role, level, identity], separators=(",", ":"))
            return hashlib.sha256(key.encode("utf-8")).digest(), identity

        return sorted(ids, key=rank)

    selected = {split: [] for split in SPLITS}
    for level in LEVELS:
        pool = ranked(
            [identity for identity, label in training.items() if label == level], "train-val", level
        )
        if len(pool) < train_per_level + val_per_level:
            raise ValueError(f"Insufficient {level} questions for disjoint train and validation sampling")
        selected["train"].extend(pool[:train_per_level])
        selected["val"].extend(pool[train_per_level : train_per_level + val_per_level])
    if len(development) < test_count:
        raise ValueError("Insufficient official development questions for the project test split")
    selected["test"] = ranked(development, "test")[:test_count]
    return selected


def _cached_sources(cache_dir: Path, revision: str) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full pinned Hugging Face dataset revision is required")
    root = cache_dir / "hotpotqa--hotpot_qa" / revision
    paths = {
        "train": sorted((root / "distractor").glob("train-*.parquet")),
        "test": sorted((root / "distractor").glob("validation-*.parquet")),
    }
    if not all(paths.values()):
        raise FileNotFoundError("Pinned train/dev Parquet cache is incomplete; no download attempted")
    for split, files in paths.items():
        prefix = "train" if split == "train" else "validation"
        matches = [re.fullmatch(rf"{prefix}-(\d+)-of-(\d+)\.parquet", path.name) for path in files]
        if not all(matches):
            raise ValueError("Unexpected HotpotQA Parquet shard filename")
        parts = [(int(match[1]), int(match[2])) for match in matches if match is not None]
        totals = {total for _, total in parts}
        if len(totals) != 1 or {index for index, _ in parts} != set(range(parts[0][1])):
            raise FileNotFoundError("Pinned HotpotQA Parquet shard set is incomplete")
    receipts = {}
    for files in paths.values():
        for path in files:
            relative = path.relative_to(root).as_posix()
            receipt = json.loads(path.with_suffix(path.suffix + ".download.json").read_text(encoding="utf-8"))
            url = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{revision}/{relative}?download=true"
            if (
                receipt.get("url") != url
                or receipt.get("bytes") != path.stat().st_size
                or receipt.get("sha256") != file_sha256(path)
            ):
                raise ValueError(f"Pinned raw dataset receipt does not match {path}")
            receipts[relative] = receipt
    return paths, receipts


def _index(paths: list[Path], *, difficulty: bool) -> dict[str, str]:
    import pyarrow.parquet as parquet

    result = {}
    for path in paths:
        source = parquet.ParquetFile(path)
        identity_field = "_id" if "_id" in source.schema_arrow.names else "id"
        columns = [identity_field, "level"] if difficulty else [identity_field]
        for batch in source.iter_batches(batch_size=4096, columns=columns):
            for record in batch.to_pylist():
                identity = record[identity_field]
                if not isinstance(identity, str) or not identity or identity in result:
                    raise ValueError("Source contains a missing or duplicate original question ID")
                result[identity] = record["level"] if difficulty else ""
    return result


def _materialize(paths: list[Path], ids: dict[str, list[str]]) -> dict[str, list[QAExample]]:
    wanted = {identity: split for split, identities in ids.items() for identity in identities}
    examples: dict[str, QAExample] = {}
    for path in paths:
        for record in iter_records(path):
            identity = record.get("_id", record.get("id"))
            if identity in wanted:
                examples[identity] = hotpotqa_example(record, wanted[identity])
    if set(examples) != set(wanted):
        raise ValueError("Selected original IDs could not all be materialized")
    return {split: [examples[identity] for identity in identities] for split, identities in ids.items()}


def _export_ids(manifest: dict[str, Any], target: Path | None) -> None:
    if target is None:
        return
    data = {f"{split}_ids": manifest["splits"][split]["ids"] for split in SPLITS}
    encoded = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise FileExistsError(f"Original ID archive differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(encoded)


def prepare_stratified_hotpotqa(
    output_dir: str | Path,
    *,
    cache_dir: str | Path,
    hf_revision: str,
    protocol_path: str | Path,
    train_per_level: int = 100,
    val_per_level: int = 50,
    test_count: int = 1000,
    seed: int = 42,
    ids_output: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare the provisional professor protocol without replacing prior splits or downloading."""
    output = Path(output_dir)
    ids_target = Path(ids_output) if ids_output is not None else None
    paths, receipts = _cached_sources(Path(cache_dir), hf_revision)
    source = {
        "kind": "official_public_stratified",
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "revision": hf_revision,
        "retrieval_backend": "revision_pinned_local_parquet",
        "raw_files": receipts,
        "official_repository": "https://github.com/hotpotqa/hotpot",
        "partition_policy": "official_train -> disjoint stratified train+val; official_validation -> test",
        "course_split_verified": False,
        "protocol_status": "provisional",
        "test_count_resolution": f"{test_count}_user_specified",
        "instruction": {"path": Path(protocol_path).name, "sha256": file_sha256(protocol_path)},
        "sampling": {
            "method": SAMPLER,
            "seed": seed,
            "difficulty_counts": {
                "train": dict.fromkeys(LEVELS, train_per_level),
                "val": dict.fromkeys(LEVELS, val_per_level),
            },
            "test_count": test_count,
            "procedure": (
                "Seeded SHA-256 rank within each original difficulty; train first, val next; dev separately"
            ),
        },
    }
    sizes = {"train": 3 * train_per_level, "val": 3 * val_per_level, "test": test_count}
    manifest_path = output / "split_manifest.json"
    if manifest_path.exists():
        manifest = verify_manifest(output)
        if (
            manifest["dataset"] != "hotpotqa"
            or manifest["seed"] != seed
            or not same_source_after_instruction_rename(manifest["source"], source)
            or {split: info["count"] for split, info in manifest["splits"].items()} != sizes
        ):
            raise FileExistsError("Existing data use a different sampling protocol; choose a new directory")
        _export_ids(manifest, ids_target)
        return manifest
    if ids_target is not None and ids_target.exists():
        raise FileExistsError(f"ID archive exists without its prepared split manifest: {ids_target}")
    training = _index(paths["train"], difficulty=True)
    development = _index(paths["test"], difficulty=False)
    ids = select_stratified_ids(
        training, list(development), train_per_level=train_per_level,
        val_per_level=val_per_level, test_count=test_count, seed=seed,
    )
    by_split = _materialize(paths["train"], {split: ids[split] for split in ("train", "val")})
    by_split.update(_materialize(paths["test"], {"test": ids["test"]}))
    manifest = freeze_splits(
        "hotpotqa", by_split, output, sizes, seed=seed, course_limits=False, source=source
    )
    _export_ids(manifest, ids_target)
    return manifest
