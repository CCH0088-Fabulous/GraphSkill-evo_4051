"""Synthetic binary fixtures exercise GGUF boundaries without real model weights."""

import hashlib
import struct

import pytest

from repro.serving.gguf import inspect_gguf, validate_gguf_file


def string(text, endian="<"):
    data = text.encode()
    return struct.pack(endian + "Q", len(data)) + data


def write_gguf(path, *, endian="<", truncate=False, extra=(), tensor_type=0):
    metadata = [
        ("general.architecture", 8, "fixture"),
        ("general.name", 8, "Synthetic test model"),
        ("fixture.context_length", 4, 8192),
        ("fixture.block_count", 4, 2),
        ("tokenizer.chat_template", 8, "{% if enable_thinking %}<think>{% endif %}"),
        *extra,
    ]
    data = b"GGUF" + struct.pack(endian + "IQQ", 3, 1, len(metadata))
    for key, kind, value in metadata:
        data += string(key, endian) + struct.pack(endian + "I", kind)
        if kind == 8:
            data += string(value, endian)
        elif kind == 9:
            data += struct.pack(endian + "IQ", 8, len(value))
            data += b"".join(string(item, endian) for item in value)
        else:
            data += struct.pack(endian + "I", value)
    data += string("embedding.weight", endian) + struct.pack(endian + "IQIQ", 1, 4, tensor_type, 0)
    data += b"\0" * ((-len(data)) % 32)
    data += b"\0" * 16
    path.write_bytes(data[:-1] if truncate else data)
    return path


@pytest.mark.parametrize("endian", ["<", ">"])
def test_inspection_and_hash_skip_token_arrays(tmp_path, endian):
    path = write_gguf(
        tmp_path / "fixture.gguf", endian=endian, extra=[("tokenizer.ggml.tokens", 9, ["a", "b", "c"])]
    )
    before = path.read_bytes()
    info = inspect_gguf(path, calculate_sha256=True)
    assert info["architecture"] == "fixture"
    assert info["context_length"] == 8192
    assert info["block_count"] == 2
    assert info["tensor_data_bounds_verified"]
    assert info["chat_template_summary"]["tokenizer.chat_template"]["has_enable_thinking"]
    assert "tokenizer.ggml.tokens" not in info["metadata"]
    assert info["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert validate_gguf_file(path)[0]


def test_truncated_tensor_payload_is_not_ready(tmp_path):
    path = write_gguf(tmp_path / "short.gguf", truncate=True)
    ready, detail = validate_gguf_file(path)
    assert not ready
    assert "truncated" in detail


def test_unknown_tensor_type_is_not_a_verified_payload(tmp_path):
    path = write_gguf(tmp_path / "unknown.gguf", tensor_type=999)
    info = inspect_gguf(path)
    assert not info["tensor_data_bounds_verified"]
    assert not validate_gguf_file(path)[0]


def test_missing_split_is_reported(tmp_path):
    path = write_gguf(tmp_path / "model-00001-of-00002.gguf", extra=[("split.count", 4, 2)])
    info = inspect_gguf(path)
    assert info["missing_shards"] == [str(tmp_path / "model-00002-of-00002.gguf")]
    assert not validate_gguf_file(path)[0]


def test_duplicate_metadata_and_bad_magic_are_rejected(tmp_path):
    path = write_gguf(tmp_path / "duplicate.gguf", extra=[("general.name", 8, "duplicate")])
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_gguf(path)
    path.write_bytes(b"bad!")
    assert not validate_gguf_file(path)[0]


def test_existing_but_corrupt_split_is_not_ready(tmp_path):
    first = write_gguf(tmp_path / "model-00001-of-00002.gguf", extra=[("split.count", 4, 2)])
    second = write_gguf(tmp_path / "model-00002-of-00002.gguf", extra=[("split.count", 4, 2)])
    assert validate_gguf_file(first)[0]
    second.write_bytes(b"GGUF")
    ready, detail = validate_gguf_file(first)
    assert not ready
    assert "Invalid GGUF shard" in detail
