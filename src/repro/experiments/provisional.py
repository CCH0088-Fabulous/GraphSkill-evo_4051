"""Validate the user-authorized HotpotQA protocol pending the teacher's size clarification."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from repro.config import load_config, resolve_path
from repro.datasets.io import load_split
from repro.datasets.manifests import verify_manifest, verify_manifest_reference
from repro.datasets.protocol import verify_instruction
from repro.experiments.formal import run_comparison_batch, validate_original_protocol

_SIZES = {"train": 300, "val": 150, "test": 1000}
_DIFFICULTIES = {
    "train": dict.fromkeys(("easy", "medium", "hard"), 100),
    "val": dict.fromkeys(("easy", "medium", "hard"), 50),
}


def load_provisional_config(
    path: str | Path, overrides: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify fixed IDs, stratification, instruction provenance and disabled test inference."""
    config = load_config(path, overrides, presets=["course", "hotpotqa.provisional"])
    if config["experiment"]["profile"] != "custom" or config["experiment"]["allow_test"]:
        raise ValueError("Provisional development requires custom profile and test disabled")
    if config["model"]["backend"] != "transformers":
        raise ValueError("Provisional HotpotQA comparisons require the shared Transformers model")
    if config["data"]["dataset"] != "hotpotqa" or not config["data"].get("manifest"):
        raise ValueError("Provisional HotpotQA requires its recorded dataset identity")
    root = resolve_path(config["data"]["split_dir"])
    manifest = verify_manifest(root)
    verify_manifest_reference(manifest, resolve_path(config["data"]["manifest"]))
    if (
        manifest["dataset"] != "hotpotqa" or manifest.get("synthetic")
        or {split: info["count"] for split, info in manifest["splits"].items()} != _SIZES
    ):
        raise ValueError("Provisional HotpotQA requires real fixed 300/150/1000 partitions")
    source = manifest.get("source", {})
    if (
        source.get("kind") != "official_public_stratified"
        or source.get("config") != "distractor"
        or source.get("protocol_status") != "provisional"
        or source.get("course_split_verified") is not False
        or source.get("test_count_resolution") != "1000_user_specified"
    ):
        raise ValueError("HotpotQA must retain its provisional stratified distractor provenance")
    sampling = source.get("sampling", {})
    if (
        sampling.get("method") != "sha256_seeded_id_rank_v1"
        or sampling.get("difficulty_counts") != _DIFFICULTIES
        or sampling.get("test_count") != 1000
        or sampling.get("seed") != manifest["seed"]
        or manifest["seed"] != config["preparation"]["seed"]
    ):
        raise ValueError("HotpotQA sampling does not match the fixed stratified protocol")
    for split, expected in _DIFFICULTIES.items():
        examples = load_split(root, split, dataset="hotpotqa")
        counts = Counter(example.metadata.get("level") for example in examples)
        if counts != expected:
            raise ValueError(f"HotpotQA {split} difficulty counts differ from the stratified protocol")
    instruction = source.get("instruction", {})
    path = resolve_path(config["preparation"]["hotpotqa_protocol_file"])
    verify_instruction(instruction, path)
    validate_original_protocol(config)
    return config, source


def run_provisional(
    path: str | Path = "config.yaml",
    overrides: list[str] | None = None,
    *,
    seeds: list[int] | None = None,
) -> Path:
    """Run only validation comparisons; the 1000 held-out questions remain unqueried."""
    config, source = load_provisional_config(path, overrides)
    return run_comparison_batch(
        {"hotpotqa": config},
        seeds=seeds,
        folder="provisional-hotpotqa",
        purpose="member_a_provisional_hotpotqa_development_comparison",
        protocol={
            "status": "provisional",
            "sizes": _SIZES,
            "instruction": source["instruction"],
            "sampling": source["sampling"],
            "test_count_resolution": source["test_count_resolution"],
            "pending_confirmation": ["scope versus PDF 100/30/50 cap"],
        },
    )
