"""Materialize SearchQA from official GraphSkillEvo ID manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from repro.contracts import QAExample
from repro.datasets.io import SPLITS, iter_records, read_records
from repro.datasets.manifests import file_sha256, freeze_splits

HF_DATASET = "lucadiliello/searchqa"


def _answer_list(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        raw = raw.get("text", raw.get("answers"))
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(value, str) for value in raw):
        return tuple(raw)
    raise ValueError("SearchQA answer must be a string or a non-empty array of strings")


def _search_context(record: Mapping[str, Any]) -> str:
    context = record.get("context")
    if isinstance(context, str):
        return context
    results = record.get("search_results", record.get("snippets", context))
    if isinstance(results, str):
        return results
    if isinstance(results, dict):
        results = results.get("snippets", results.get("snippet"))
    if isinstance(results, (list, tuple)):
        snippets = []
        for item in results:
            if isinstance(item, str):
                snippets.append(item)
            elif isinstance(item, dict) and isinstance(item.get("snippet"), str):
                snippets.append(item["snippet"])
            else:
                raise ValueError("Unsupported SearchQA search-result format")
        return "\n".join(snippets)
    raise ValueError("SearchQA source record is missing context or search-result snippets")


def searchqa_example(record: dict[str, Any], split: str) -> QAExample:
    identity = record.get("key", record.get("id"))
    if not isinstance(identity, str) or not identity:
        raise ValueError("SearchQA source record requires a non-empty key or id")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"SearchQA {identity}: missing question")
    metadata = dict(record.get("metadata", {}))
    metadata.update({"source_record": record, "source_id_field": "key"})
    return QAExample(
        id=identity,
        dataset="searchqa",
        split=split,
        question=question,
        context=_search_context(record),
        answers=_answer_list(record.get("answers", record.get("answer"))),
        metadata=metadata,
    )


def _manifest_ids(manifest_dir: Path) -> dict[str, list[str]]:
    by_split = {}
    seen: set[str] = set()
    for split in SPLITS:
        records = read_records(manifest_dir / split / "items.json")
        identities = []
        for record in records:
            identity = record.get("id")
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{split} manifest contains an invalid ID")
            if identity in seen:
                raise ValueError(f"Official manifest contains overlapping or duplicate ID {identity!r}")
            seen.add(identity)
            identities.append(identity)
        by_split[split] = identities
    return by_split


def _local_records(raw_path: Path) -> Iterable[dict[str, Any]]:
    if raw_path.is_file():
        yield from iter_records(raw_path)
        return
    paths = sorted([*raw_path.rglob("*.json"), *raw_path.rglob("*.jsonl"), *raw_path.rglob("*.parquet")])
    paths = [path for path in paths if not path.name.endswith(".download.json")]
    if not paths:
        raise FileNotFoundError(f"{raw_path}: no source JSON, JSONL, or Parquet files")
    for path in paths:
        yield from iter_records(path)


def _hf_records(revision: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise RuntimeError("HF data support requires: pip install -e '.[data]'") from exc
    for split in get_dataset_split_names(HF_DATASET, revision=revision):
        yield from load_dataset(HF_DATASET, split=split, revision=revision, streaming=True)


def prepare_searchqa(
    raw_path: str | Path | None,
    manifest_dir: str | Path,
    output_dir: str | Path,
    sizes: Mapping[str, int] | None = None,
    *,
    seed: int = 42,
    hf_revision: str | None = None,
    download_mode: str = "parquet",
    cache_dir: str | Path = "data/raw",
    course_limits: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select fixed IDs first, then find those records locally or in the HF dataset.

    Passing raw_path=None explicitly opts into dataset download. No model is loaded.
    Supplied GraphSkillEvo partition boundaries take precedence over upstream splits.
    """
    sizes = dict(sizes if sizes is not None else {"train": 100, "val": 30, "test": 50})
    if download_mode not in {"viewer", "streaming", "parquet"}:
        raise ValueError("download_mode must be viewer, streaming, or parquet")
    if hf_revision is not None and download_mode == "viewer":
        raise ValueError("Dataset Viewer does not support revision pinning; select download_mode='streaming'")
    manifest_root = Path(manifest_dir)
    identities = _manifest_ids(manifest_root)
    if set(sizes).difference(SPLITS):
        raise ValueError("Subset sizes accept only train, val, and test keys")
    wanted: dict[str, str] = {}
    for split in SPLITS:
        count = sizes.get(split, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid {split} subset size")
        if count > len(identities[split]):
            raise ValueError(f"{split}: requested {count}, manifest contains {len(identities[split])}")
        if course_limits and count > {"train": 100, "val": 30, "test": 50}[split]:
            raise ValueError(f"{split}: requested size exceeds the course profile")

        def rank(identity: str, partition: str = split) -> tuple[str, str]:
            key = f"{seed}\0searchqa\0{partition}\0{identity}".encode()
            return hashlib.sha256(key).hexdigest(), identity

        for identity in sorted(identities[split], key=rank)[:count]:
            wanted[identity] = split
    source: dict[str, Any] = {
        "dataset": HF_DATASET,
        "official_manifest_dir": str(manifest_root.resolve()),
        "official_manifest_sha256": {
            split: file_sha256(manifest_root / split / "items.json") for split in SPLITS
        },
    }
    if raw_path is None and download_mode == "parquet":
        from repro.datasets.downloads import download_hf_parquet

        paths, retrieval = download_hf_parquet(HF_DATASET, cache_dir, prefix="data/", revision=hf_revision)
        source.update(retrieval)
        records = (record for path in paths for record in iter_records(path))
    elif raw_path is None and download_mode == "viewer":
        from repro.datasets.hub import viewer_records_by_ids

        records, retrieval = viewer_records_by_ids(HF_DATASET, sorted(wanted))
        source.update(retrieval)
    elif raw_path is None:
        if hf_revision is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError("HF data support requires: pip install -e '.[data]'") from exc
            hf_revision = HfApi().dataset_info(HF_DATASET).sha
        if not hf_revision:
            raise ValueError("Could not resolve a fixed Hugging Face dataset revision")
        source["revision"] = hf_revision
        source["retrieval_backend"] = "revision_pinned_hf_streaming"
        records = _hf_records(hf_revision)
    else:
        raw = Path(raw_path)
        if not raw.exists():
            raise FileNotFoundError(raw)
        source["raw_path"] = str(raw.resolve())
        source["raw_sha256"] = (
            file_sha256(raw)
            if raw.is_file()
            else {
                str(path.relative_to(raw)): file_sha256(path)
                for path in sorted([*raw.rglob("*.json"), *raw.rglob("*.jsonl")])
            }
        )
        records = _local_records(raw)
    by_split: dict[str, list[QAExample]] = {split: [] for split in SPLITS}
    found: set[str] = set()
    for record in records:
        identity = record.get("key", record.get("id"))
        if identity not in wanted:
            continue
        if identity in found:
            raise ValueError(f"Source contains duplicate requested ID {identity!r}")
        found.add(identity)
        by_split[wanted[identity]].append(searchqa_example(record, wanted[identity]))
    missing = set(wanted).difference(found)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Source is missing {len(missing)} selected manifest IDs; first: {preview}")
    manifest = freeze_splits(
        "searchqa",
        by_split,
        output_dir,
        sizes,
        seed=seed,
        course_limits=course_limits,
        source=source,
        overwrite=overwrite,
    )
    return manifest


def official_split_payload(split_dir: str | Path, output_dir: str | Path) -> None:
    """Export normalized records to the unmodified upstream directory layout."""
    from repro.datasets.io import load_split

    root = Path(output_dir)
    for split in SPLITS:
        target = root / split / "items.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        records = [example.to_dict() for example in load_split(split_dir, split, dataset="searchqa")]
        target.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
