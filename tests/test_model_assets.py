from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ASSETS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_model_assets.py"))


@pytest.mark.parametrize(
    "filename",
    [
        "model.safetensors",
        "model-00001-of-00004.safetensors",
        "pytorch_model.bin",
        "model.pt",
        "model.gguf",
        "../config.json",
        "subfolder/tokenizer.json",
        "unlisted.json",
    ],
)
def test_weights_and_non_allowlisted_names_are_rejected(filename):
    assert not ASSETS["is_allowed_asset"](filename)


def test_weight_index_is_allowed_but_weight_payload_is_not():
    assert ASSETS["is_allowed_asset"]("model.safetensors.index.json")
    siblings = [SimpleNamespace(rfilename=name, size=10) for name in ASSETS["REQUIRED_ASSETS"]]
    siblings += [SimpleNamespace(rfilename="model-00001-of-00004.safetensors", size=8_000_000_000)]
    selected = ASSETS["select_assets"](siblings)
    assert set(selected) == ASSETS["REQUIRED_ASSETS"]


def test_download_revision_must_be_immutable():
    with pytest.raises(ValueError, match="immutable"):
        ASSETS["validate_revision"]("main")
    assert ASSETS["validate_revision"](ASSETS["PINNED_REVISION"]) == ASSETS["PINNED_REVISION"]


def test_allowlisted_asset_size_is_bounded():
    siblings = [SimpleNamespace(rfilename=name, size=10) for name in ASSETS["REQUIRED_ASSETS"]]
    siblings.append(SimpleNamespace(rfilename="vocab.json", size=ASSETS["MAX_FILE_BYTES"] + 1))
    with pytest.raises(ValueError, match="limit"):
        ASSETS["select_assets"](siblings)


def test_existing_different_revision_is_not_overwritten(tmp_path, monkeypatch):
    import huggingface_hub

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("Revision mismatch must fail before any network query")

    monkeypatch.setattr(huggingface_hub, "HfApi", unexpected_network)
    (tmp_path / "asset_manifest.json").write_text(
        json.dumps(
            {
                "repository": ASSETS["REPOSITORY"],
                "revision": "0" * 40,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different asset revision"):
        ASSETS["prepare_assets"](tmp_path, ASSETS["PINNED_REVISION"])


def test_local_recheck_rejects_changed_asset_before_loading_tokenizer(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    (tmp_path / "asset_manifest.json").write_text(
        json.dumps(
            {
                "repository": ASSETS["REPOSITORY"],
                "revision": ASSETS["PINNED_REVISION"],
                "files": [{"file": "config.json", "size_bytes": 2, "sha256": "invalid"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        ASSETS["verify_local_assets"](tmp_path, ASSETS["PINNED_REVISION"])
