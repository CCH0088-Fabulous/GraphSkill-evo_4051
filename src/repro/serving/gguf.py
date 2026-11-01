"""Read GGUF identities and tensor bounds without loading or changing model tensors."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}
# GGML block size and encoded byte size; unsupported new types are reported, never guessed.
_TENSOR_BLOCKS = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    30: (1, 2),
}
_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    30: "BF16",
}
_KEEP = {
    "general.name",
    "general.basename",
    "general.architecture",
    "general.size_label",
    "general.file_type",
    "general.quantization_version",
    "general.alignment",
    "general.finetune",
    "general.version",
    "general.source.url",
    "general.source.huggingface.repository",
    "split.no",
    "split.count",
    "split.tensors.count",
}
_MODEL_SUFFIXES = (
    ".context_length",
    ".block_count",
    ".embedding_length",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.key_length",
    ".attention.value_length",
)


class _Reader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream, self.size, self.endian = stream, size, "<"

    def read(self, length: int) -> bytes:
        if length < 0 or self.stream.tell() + length > self.size:
            raise ValueError("GGUF is truncated or contains an out-of-file length")
        value = self.stream.read(length)
        if len(value) != length:
            raise ValueError("GGUF is truncated")
        return value

    def skip(self, length: int) -> None:
        if length < 0 or self.stream.tell() + length > self.size:
            raise ValueError("GGUF metadata points beyond the file")
        self.stream.seek(length, 1)

    def number(self, fmt: str) -> Any:
        return struct.unpack(self.endian + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, *, capture: bool, max_bytes: int = 4 * 1024 * 1024) -> str | None:
        length = self.number("Q")
        if capture:
            if length > max_bytes:
                raise ValueError("GGUF captured string exceeds the inspection limit")
            try:
                return self.read(length).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("GGUF metadata string is not UTF-8") from exc
        self.skip(length)
        return None

    def value(self, kind: int, *, capture: bool, depth: int = 0) -> Any:
        if kind in _SCALARS:
            value = self.number(_SCALARS[kind])
            if kind == 7:
                if value not in (0, 1):
                    raise ValueError("Invalid GGUF boolean")
                value = bool(value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite GGUF metadata number")
            return value if capture else None
        if kind == 8:
            return self.string(capture=capture)
        if kind == 9:
            if depth >= 16:
                raise ValueError("GGUF nested array depth exceeds the inspection limit")
            element_type, count = self.number("I"), self.number("Q")
            if count > 10_000_000:
                raise ValueError("GGUF array count exceeds the inspection limit")
            if element_type in _SCALARS and element_type != 7:
                self.skip(struct.calcsize(_SCALARS[element_type]) * count)
            else:
                for _ in range(count):
                    self.value(element_type, capture=False, depth=depth + 1)
            return {"array_type": element_type, "count": count} if capture else None
        raise ValueError(f"Unknown GGUF metadata value type: {kind}")


def _template_summary(template: str) -> dict[str, Any]:
    return {
        "length_chars": len(template),
        "sha256": hashlib.sha256(template.encode()).hexdigest(),
        "has_enable_thinking": "enable_thinking" in template,
        "has_think_tags": "<think>" in template or "</think>" in template,
        "has_reasoning_effort": "reasoning_effort" in template,
        "thinking_condition_excerpts": [
            match.group(0)[:240] for match in re.finditer(r"\{%[^%]*(?:thinking|reasoning)[^%]*%\}", template)
        ][:8],
    }


def _shards(path: Path, metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    count = metadata.get("split.count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10000:
        raise ValueError("Invalid GGUF shard count")
    if count == 1:
        return [str(path)], []
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name, re.IGNORECASE)
    if match is None or int(match[3]) != count:
        raise ValueError("Split GGUF filename does not match its declared shard count")
    expected = [
        path.with_name(f"{match[1]}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)
    ]
    return [str(item) for item in expected], [str(item) for item in expected if not item.is_file()]


def inspect_gguf(path: str | Path, *, calculate_sha256: bool = False) -> dict[str, Any]:
    """Inspect one local GGUF, skipping token arrays and never mapping tensor data.

    Tensor offsets and byte counts are checked for known GGML types. This is a
    structural check, not a numerical validation of tensor contents or proof that
    a particular inference engine implements the architecture.
    """
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".gguf" or not source.is_file():
        raise ValueError("GGUF model must be an existing local .gguf file")
    before = source.stat()
    metadata: dict[str, Any] = {}
    templates: dict[str, Any] = {}
    tensor_types: Counter[int] = Counter()
    unknown_types: set[int] = set()
    ranges = []
    with source.open("rb") as stream:
        reader = _Reader(stream, before.st_size)
        if reader.read(4) != b"GGUF":
            raise ValueError("Invalid GGUF magic")
        version_bytes = reader.read(4)
        version = struct.unpack("<I", version_bytes)[0]
        if version not in (2, 3):
            version = struct.unpack(">I", version_bytes)[0]
            reader.endian = ">"
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
        if not 1 <= tensor_count <= 1_000_000 or not 1 <= metadata_count <= 100_000:
            raise ValueError("Invalid or excessive GGUF tensor/metadata count")
        seen = set()
        for _ in range(metadata_count):
            key = reader.string(capture=True, max_bytes=65535)
            if key in seen:
                raise ValueError(f"Duplicate GGUF metadata key: {key}")
            seen.add(key)
            kind = reader.number("I")
            keep = key in _KEEP or key.endswith(_MODEL_SUFFIXES)
            is_template = key == "tokenizer.chat_template" or key.startswith("tokenizer.chat_template.")
            value = reader.value(kind, capture=keep or is_template)
            if keep:
                metadata[key] = value
            elif is_template and isinstance(value, str):
                templates[key] = _template_summary(value)
            if stream.tell() > 256 * 1024 * 1024:
                raise ValueError("GGUF metadata exceeds the inspection limit")
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0 or alignment > 1024 * 1024:
            raise ValueError("Invalid GGUF alignment")
        names = set()
        for _ in range(tensor_count):
            name = reader.string(capture=True, max_bytes=65535)
            if name in names:
                raise ValueError(f"Duplicate GGUF tensor name: {name}")
            names.add(name)
            rank = reader.number("I")
            if not 1 <= rank <= 4:
                raise ValueError("Invalid GGUF tensor rank")
            dimensions = [reader.number("Q") for _ in range(rank)]
            if any(size == 0 for size in dimensions):
                raise ValueError("Invalid zero-sized GGUF tensor")
            kind, offset = reader.number("I"), reader.number("Q")
            tensor_types[kind] += 1
            if offset % alignment:
                raise ValueError("GGUF tensor offset is not aligned")
            if kind in _TENSOR_BLOCKS:
                block, size = _TENSOR_BLOCKS[kind]
                if dimensions[0] % block:
                    raise ValueError("GGUF tensor row does not fit its quantization block")
                length = math.prod(dimensions) // block * size
            else:
                unknown_types.add(kind)
                length = 1
            ranges.append((offset, offset + length))
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        for _start, end in ranges:
            if data_start + end > before.st_size:
                raise ValueError("GGUF tensor payload extends beyond the file; model is truncated")
        ordered = sorted(ranges)
        if any(current[0] < previous[1] for previous, current in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("GGUF tensor payload ranges overlap")
        digest = None
        if calculate_sha256:
            stream.seek(0)
            sha = hashlib.sha256()
            while chunk := stream.read(8 * 1024 * 1024):
                sha.update(chunk)
            digest = sha.hexdigest()
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError("GGUF file changed while it was being inspected")
    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("GGUF general.architecture is missing")
    shards, missing_shards = _shards(source, metadata)
    return {
        "path": str(source),
        "format": "GGUF",
        "version": version,
        "byte_order": "little" if reader.endian == "<" else "big",
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "file_size_bytes": before.st_size,
        "file_size_gib": round(before.st_size / 1024**3, 3),
        "metadata": metadata,
        "quantization": "Q4_K_M" if metadata.get("general.file_type") == 15 else None,
        "architecture": architecture,
        "context_length": metadata.get(f"{architecture}.context_length"),
        "block_count": metadata.get(f"{architecture}.block_count"),
        "chat_template_summary": templates,
        "tensor_data_offset": data_start,
        "tensor_type_counts": {
            _TYPE_NAMES.get(kind, str(kind)): count for kind, count in tensor_types.items()
        },
        "unknown_tensor_types": sorted(unknown_types),
        "tensor_data_bounds_verified": not unknown_types,
        "expected_shards": shards,
        "missing_shards": missing_shards,
        "sha256": digest,
        "weights_loaded": False,
        "validation_scope": (
            "Header, metadata, tensor descriptors and known-type byte bounds; no tensor values decoded"
        ),
    }


def validate_gguf_file(path: str | Path) -> tuple[bool, str]:
    try:
        info = inspect_gguf(path)
    except (OSError, ValueError, struct.error) as exc:
        return False, str(exc)
    if info["missing_shards"]:
        return False, f"Missing GGUF shards: {', '.join(info['missing_shards'])}"
    if not info["tensor_data_bounds_verified"]:
        return False, f"Cannot verify byte bounds for GGML types: {info['unknown_tensor_types']}"
    for index, shard in enumerate(info["expected_shards"]):
        if Path(shard) == Path(info["path"]):
            shard_info = info
        else:
            try:
                shard_info = inspect_gguf(shard)
            except (OSError, ValueError, struct.error) as exc:
                return False, f"Invalid GGUF shard {shard}: {exc}"
        if not shard_info["tensor_data_bounds_verified"]:
            return False, f"Cannot verify tensor bounds in GGUF shard: {shard}"
        if shard_info["architecture"] != info["architecture"]:
            return False, f"GGUF shard architecture differs: {shard}"
        shard_meta = shard_info["metadata"]
        if shard_meta.get("split.count", 1) != len(info["expected_shards"]):
            return False, f"GGUF shard count differs: {shard}"
        if "split.no" in shard_meta and shard_meta["split.no"] != index:
            return False, f"GGUF shard index differs: {shard}"
    return True, (
        f"GGUF v{info['version']} {info['architecture']}; {info['file_size_gib']} GiB; "
        "tensor byte bounds verified, inference not implied"
    )
