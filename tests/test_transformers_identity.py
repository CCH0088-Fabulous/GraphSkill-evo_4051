"""Weight and tokenizer changes must alter the identity of a formal model run."""

import json

from repro.artifacts.transformers_identity import fingerprint_directory


def test_model_fingerprint_tracks_weights_and_tokenizer(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "part.safetensors").write_bytes(b"fixture weights, not a real model")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "part.safetensors"}}), encoding="utf-8",
    )
    first = fingerprint_directory(tmp_path)
    (tmp_path / "part.safetensors").write_bytes(b"different fixture weights")
    second = fingerprint_directory(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text('{"changed": true}', encoding="utf-8")
    third = fingerprint_directory(tmp_path)
    assert len({row["fingerprint_sha256"] for row in (first, second, third)}) == 3
    assert first["source_revision"] is None
