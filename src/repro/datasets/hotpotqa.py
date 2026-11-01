"""Adapt course-provided HotpotQA files without inventing new split boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from repro.contracts import QAExample
from repro.datasets.io import SPLITS, read_records, select_parquet_rows
from repro.datasets.manifests import file_sha256, freeze_splits


def format_context(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        titles = context.get("title")
        sentences = context.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentences, list) or len(titles) != len(sentences):
            raise ValueError("Invalid Hugging Face HotpotQA title/sentences columns")
        context = list(zip(titles, sentences, strict=True))
    if not isinstance(context, list):
        raise ValueError("HotpotQA context must be a string or [title, sentences] array")
    paragraphs = []
    for block in context:
        if isinstance(block, dict):
            title = block.get("title")
            sentences = block.get("sentences")
        elif isinstance(block, (list, tuple)) and len(block) == 2:
            title, sentences = block
        else:
            raise ValueError("Invalid HotpotQA context block")
        if not isinstance(title, str):
            raise ValueError("HotpotQA context title must be a string")
        if not isinstance(sentences, list) or not all(isinstance(sentence, str) for sentence in sentences):
            raise ValueError("HotpotQA context sentences must be a string array")
        paragraphs.append(f"[{title}]\n" + " ".join(sentence.strip() for sentence in sentences))
    return "\n\n".join(paragraphs)


def hotpotqa_example(record: dict[str, Any], split: str) -> QAExample:
    identity = record.get("_id", record.get("id"))
    if not isinstance(identity, str) or not identity:
        raise ValueError("HotpotQA record requires a non-empty _id or id")
    supplied_split = record.get("split")
    if supplied_split is not None and supplied_split != split:
        raise ValueError(f"HotpotQA {identity}: source split {supplied_split!r} differs from {split!r}")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"HotpotQA {identity}: missing question")
    answers = record.get("answers", record.get("answer"))
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, (list, tuple)) or not answers:
        raise ValueError(f"HotpotQA {identity}: reference answers are required")
    if not all(isinstance(answer, str) for answer in answers):
        raise ValueError(f"HotpotQA {identity}: reference answers must be strings")
    metadata = dict(record.get("metadata", {}))
    metadata["source_record"] = record
    for name in ("type", "level", "supporting_facts"):
        if name in record:
            metadata[name] = record[name]
    return QAExample(
        id=identity,
        dataset="hotpotqa",
        split=split,
        question=question,
        context=format_context(record.get("context")),
        answers=tuple(answers),
        metadata=metadata,
    )


def prepare_hotpotqa(
    split_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    sizes: Mapping[str, int] | None = None,
    *,
    seed: int = 42,
    course_limits: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Each path must already belong to its declared course train/val/test split.

    This function deliberately offers no automatic split or public-data substitution.
    A missing official test file is allowed only when its requested subset size is zero.
    """
    if set(split_paths).difference(SPLITS):
        raise ValueError("HotpotQA source paths accept only train, val, and test")
    sizes = dict(sizes if sizes is not None else {"train": 100, "val": 30, "test": 50})
    by_split = {}
    source = {"kind": "course_provided", "files": {}}
    for split in SPLITS:
        path_value = split_paths.get(split)
        if path_value is None:
            if sizes.get(split, 0):
                raise ValueError(f"Course-provided HotpotQA {split} file is required")
            by_split[split] = []
            continue
        path = Path(path_value)
        by_split[split] = [hotpotqa_example(record, split) for record in read_records(path)]
        source["files"][split] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return freeze_splits(
        "hotpotqa",
        by_split,
        output_dir,
        sizes,
        seed=seed,
        course_limits=course_limits,
        source=source,
        overwrite=overwrite,
    )


def prepare_public_hotpotqa(
    output_dir: str | Path,
    sizes: Mapping[str, int] | None = None,
    *,
    seed: int = 42,
    raw_train_path: str | Path | None = None,
    raw_dev_path: str | Path | None = None,
    download_mode: str = "parquet",
    cache_dir: str | Path = "data/raw",
    hf_revision: str | None = None,
    course_limits: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare a disclosed custom subset from the official public distractor data.

    Official train supplies disjoint optimization and validation subsets. Official dev
    supplies a local held-out test subset. This is NOT a verified course-provided split.
    Row offsets are fixed before inspecting selected records. Parquet mode preserves
    the same row-offset selection as the optional Dataset Viewer mode.
    """
    sizes = dict(sizes if sizes is not None else {"train": 100, "val": 30, "test": 50})
    if download_mode not in {"parquet", "viewer"}:
        raise ValueError("download_mode must be parquet or viewer")
    if hf_revision and download_mode == "viewer":
        raise ValueError("Dataset Viewer cannot pin a historical revision; use parquet mode")
    if set(sizes).difference(SPLITS):
        raise ValueError("Subset sizes accept only train, val, and test keys")
    for split in SPLITS:
        count = sizes.get(split, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid {split} subset size")
        if course_limits and count > {"train": 100, "val": 30, "test": 50}[split]:
            raise ValueError(f"{split}: requested size exceeds the course profile")
    train_count = sizes.get("train", 0)
    val_count = sizes.get("val", 0)
    test_count = sizes.get("test", 0)
    source: dict[str, Any] = {
        "kind": "official_public_custom_split",
        "official_repository": "https://github.com/hotpotqa/hotpot",
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "course_split_verified": False,
        "partition_policy": "official_train -> disjoint local train+val; official_validation -> local test",
        "heldout_source": "official_validation",
    }
    parquet_paths = []
    if download_mode == "parquet" and (raw_train_path is None or (raw_dev_path is None and test_count)):
        from repro.datasets.downloads import download_hf_parquet

        parquet_paths, retrieval = download_hf_parquet(
            "hotpotqa/hotpot_qa",
            cache_dir,
            prefix="distractor/",
            revision=hf_revision,
        )
        source.update(retrieval)
    if raw_train_path is None and download_mode == "parquet":
        train_records, train_source = select_parquet_rows(
            [path for path in parquet_paths if path.name.startswith("train-")],
            train_count + val_count,
            seed=seed,
        )
        source["training_retrieval"] = train_source
    elif raw_train_path is None:
        from repro.datasets.hub import viewer_random_rows

        train_records, train_source = viewer_random_rows(
            "hotpotqa/hotpot_qa",
            "distractor",
            "train",
            train_count + val_count,
            seed=seed,
        )
        source["training_retrieval"] = train_source
    else:
        path = Path(raw_train_path)
        train_records = read_records(path)
        source["training_retrieval"] = {"file": str(path.resolve()), "sha256": file_sha256(path)}
    if raw_dev_path is None and test_count and download_mode == "parquet":
        test_records, test_source = select_parquet_rows(
            [path for path in parquet_paths if path.name.startswith("validation-")],
            test_count,
            seed=seed + 1,
        )
        source["test_retrieval"] = test_source
    elif raw_dev_path is None and test_count:
        from repro.datasets.hub import viewer_random_rows

        test_records, test_source = viewer_random_rows(
            "hotpotqa/hotpot_qa",
            "distractor",
            "validation",
            test_count,
            seed=seed + 1,
        )
        source["test_retrieval"] = test_source
    elif raw_dev_path is not None:
        path = Path(raw_dev_path)
        test_records = read_records(path)
        source["test_retrieval"] = {"file": str(path.resolve()), "sha256": file_sha256(path)}
    else:
        test_records = []

    def rank(record: dict[str, Any]) -> tuple[str, str]:
        identity = record.get("_id", record.get("id"))
        if not isinstance(identity, str) or not identity:
            raise ValueError("HotpotQA source record requires a non-empty id")
        return hashlib.sha256(f"{seed}\0public-hotpotqa\0{identity}".encode()).hexdigest(), identity

    ranked_train = sorted(train_records, key=rank)
    if len(ranked_train) < train_count + val_count:
        raise ValueError("Official train source has too few records for disjoint local train/val subsets")
    selected_train = ranked_train[:train_count]
    selected_val = ranked_train[train_count : train_count + val_count]
    selected_test = sorted(test_records, key=rank)[:test_count]
    if len(selected_test) != test_count:
        raise ValueError("Official dev source has too few records for the local held-out subset")
    by_split = {
        "train": [hotpotqa_example(record, "train") for record in selected_train],
        "val": [hotpotqa_example(record, "val") for record in selected_val],
        "test": [hotpotqa_example(record, "test") for record in selected_test],
    }
    for split, examples in by_split.items():
        by_split[split] = [
            replace(
                example,
                metadata={
                    **example.metadata,
                    "course_split_verified": False,
                    "upstream_split": "validation" if split == "test" else "train",
                },
            )
            for example in examples
        ]
    manifest = freeze_splits(
        "hotpotqa",
        by_split,
        output_dir,
        sizes,
        seed=seed,
        course_limits=course_limits,
        source=source,
        overwrite=overwrite,
    )
    manifest["course_split_verified"] = False
    (Path(output_dir) / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
