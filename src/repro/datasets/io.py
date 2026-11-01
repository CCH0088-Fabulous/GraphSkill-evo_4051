"""Read and validate normalized QA records and local source files."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from repro.contracts import QAExample

SPLITS = ("train", "val", "test")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array, a {data: [...]} object, or a JSONL source file."""
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if source.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON; use .jsonl for JSON Lines files") from exc
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            records = data["data"]
        elif isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
            records = list(data.values())
        else:
            raise ValueError(f"{source}: expected a JSON array, a data array, or a map of records")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"{source}: each record must be an object")
    return records


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream Parquet in small batches, or read a local JSON/JSONL source."""
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet support requires: pip install -e '.[data]'") from exc
        for batch in parquet.ParquetFile(source).iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
    yield from read_records(source)


def select_parquet_rows(
    paths: list[Path],
    count: int,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve predetermined row offsets from ordered local Parquet shards."""
    import pyarrow.parquet as parquet

    ordered = sorted(paths)
    total = sum(parquet.ParquetFile(path).metadata.num_rows for path in ordered)
    if count < 0 or count > total:
        raise ValueError(f"Requested {count} rows, Parquet source contains {total}")
    offsets = sorted(random.Random(seed).sample(range(total), count))
    wanted = set(offsets)
    records = []
    position = 0
    for path in ordered:
        for record in iter_records(path):
            if position in wanted:
                records.append(record)
            position += 1
    if len(records) != count:
        raise ValueError("Could not materialize all predetermined Parquet row offsets")
    return records, {
        "source_rows": total,
        "selected_row_offsets": offsets,
        "seed": seed,
        "row_offset_selection": "random.Random(seed).sample before record inspection",
    }


def example_from_record(record: dict[str, Any], *, source: str = "record") -> QAExample:
    """Validate the normalized on-disk contract without silently coercing bad fields."""
    required = ("id", "dataset", "split", "question", "context", "answers")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"{source}: missing fields: {', '.join(missing)}")
    for name in ("id", "dataset", "question"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{source}: {name} must be a non-empty string")
    if record["dataset"] not in {"searchqa", "hotpotqa"}:
        raise ValueError(f"{source}: unsupported dataset {record['dataset']!r}")
    if record["split"] not in SPLITS:
        raise ValueError(f"{source}: split must be train, val, or test")
    if not isinstance(record["context"], str):
        raise ValueError(f"{source}: context must be a string")
    answers = record["answers"]
    if not isinstance(answers, (list, tuple)) or not answers:
        raise ValueError(f"{source}: answers must be a non-empty string array")
    if not all(isinstance(answer, str) for answer in answers):
        raise ValueError(f"{source}: every answer must be a string")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: metadata must be an object")
    return QAExample(
        id=record["id"],
        dataset=record["dataset"],
        split=record["split"],
        question=record["question"],
        context=record["context"],
        answers=tuple(answers),
        metadata=metadata,
    )


def load_examples(path: str | Path) -> list[QAExample]:
    examples = [
        example_from_record(record, source=f"{path}:record {index}")
        for index, record in enumerate(read_records(path), 1)
    ]
    ids: set[tuple[str, str]] = set()
    for example in examples:
        identity = (example.dataset, example.id)
        if identity in ids:
            raise ValueError(f"{path}: duplicate example ID {example.id!r}")
        ids.add(identity)
    return examples


def load_split(split_dir: str | Path, split: str, dataset: str | None = None) -> list[QAExample]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    root = Path(split_dir)
    candidates = (root / f"{split}.jsonl", root / split / "items.json")
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"Missing normalized split: {root / f'{split}.jsonl'}")
    examples = load_examples(source)
    for example in examples:
        if example.split != split:
            raise ValueError(f"{source}: record {example.id!r} belongs to {example.split}, not {split}")
        if dataset is not None and example.dataset != dataset:
            raise ValueError(f"{source}: record {example.id!r} belongs to {example.dataset}, not {dataset}")
    return examples


def write_examples(path: str | Path, examples: Iterable[QAExample]) -> None:
    target = Path(path)
    records = list(examples)
    for index, example in enumerate(records, 1):
        example_from_record(example.to_dict(), source=f"output record {index}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for example in records:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
