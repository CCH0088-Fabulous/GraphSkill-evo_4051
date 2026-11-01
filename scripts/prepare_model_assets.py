"""Prepare only allowlisted tokenizer/config assets from a pinned Qwen revision; never download weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = "Qwen/Qwen3.5-9B"
# Immutable commit from official metadata inspection; branch names are not accepted for downloads.
PINNED_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ASSETS = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    }
)
REQUIRED_ASSETS = frozenset({"config.json", "tokenizer_config.json", "tokenizer.json"})
FORBIDDEN_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"})
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024


def validate_revision(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Downloads require an immutable 40-character hexadecimal commit, not a branch or tag"
        )
    return revision


def is_allowed_asset(filename: str) -> bool:
    return filename in ALLOWED_ASSETS and Path(filename).suffix.lower() not in FORBIDDEN_SUFFIXES


def select_assets(siblings: Sequence[Any]) -> list[str]:
    selected: list[str] = []
    total_size = 0
    for sibling in siblings:
        filename = sibling.rfilename
        if not is_allowed_asset(filename):
            continue
        size = sibling.size
        if not isinstance(size, int) or size < 0 or size > MAX_FILE_BYTES:
            raise ValueError(f"Asset size is missing or exceeds the configured limit: {filename}")
        total_size += size
        selected.append(filename)
    if total_size > MAX_TOTAL_BYTES:
        raise ValueError("Allowlisted assets exceed the configured total-size limit")
    if missing := REQUIRED_ASSETS.difference(selected):
        raise ValueError(f"Repository is missing required tokenizer/config assets: {sorted(missing)}")
    return sorted(selected)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tokenizer_assets(destination: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Use only the local tokenizer and pure prompt builders; no model is loaded."""
    from graphskillevo.envs.searchqa.rollout import _build_system, _build_user
    from graphskillevo.operators import EvolutionOperators
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(destination), local_files_only=True, trust_remote_code=False
    )
    initial_skill = (
        project_root / "third_party/GraphSkillEvo/graphskillevo/initial_skills/searchqa/init_skill.md"
    ).read_text(encoding="utf-8")
    question = "What is the capital of France?"
    context = "Paris is the capital of France."
    qa_user = _build_user(question, context)
    operators = EvolutionOperators(
        env_name="searchqa",
        max_completion_tokens=4096,
        reasoning_effort=None,
    )
    optimizer_system, optimizer_user = operators._build_mutation_prompt(
        scope="graph",
        parent=initial_skill,
        context=(
            "Synthetic training failure: reference answer Paris; prediction Lyon. Check supporting context."
        ),
    )
    prompts = {
        "qa_no_skill": [
            {"role": "system", "content": _build_system("")},
            {"role": "user", "content": qa_user},
        ],
        "qa_initial_skill": [
            {"role": "system", "content": _build_system(initial_skill)},
            {"role": "user", "content": qa_user},
        ],
        "optimizer_graph_mutation_one_synthetic_failure": [
            {"role": "system", "content": optimizer_system},
            {"role": "user", "content": optimizer_user},
        ],
    }
    counts = {}
    for name, messages in prompts.items():
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokens["input_ids"]
        counts[name] = len(input_ids[0]) if input_ids and isinstance(input_ids[0], list) else len(input_ids)
    disabled = tokenizer.apply_chat_template(
        prompts["qa_no_skill"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    enabled = tokenizer.apply_chat_template(
        prompts["qa_no_skill"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if disabled == enabled:
        raise ValueError("The downloaded chat template did not respond to enable_thinking")
    return {
        "local_files_only": True,
        "model_weights_loaded": False,
        "string_message_content_supported": True,
        "enable_thinking": False,
        "thinking_flag_changes_template": True,
        "sample_input_tokens": counts,
        "sample_note": (
            "Short synthetic question/context and one synthetic failure; not worst-case prompt lengths"
        ),
    }


def verify_local_assets(destination: Path, revision: str) -> dict[str, Any]:
    """Recheck every prepared file and the tokenizer without accessing the hub."""
    revision = validate_revision(revision)
    manifest_path = destination / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != revision:
        raise ValueError("Local assets do not match the requested pinned repository revision")
    filenames = set()
    for entry in manifest["files"]:
        filename = entry["file"]
        if not is_allowed_asset(filename):
            raise ValueError(f"Forbidden file in local asset manifest: {filename}")
        path = destination / filename
        if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Local asset checksum or size mismatch: {filename}")
        filenames.add(filename)
    if missing := REQUIRED_ASSETS.difference(filenames):
        raise ValueError(f"Missing required tokenizer/config assets: {sorted(missing)}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest["tokenizer_check"] = verify_tokenizer_assets(destination)
    manifest["validated_at_utc"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_assets(destination: Path, revision: str) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    revision = validate_revision(revision)
    destination = destination.expanduser().resolve()
    manifest_path = destination / "asset_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("revision") != revision or previous.get("repository") != REPOSITORY:
            raise ValueError("Destination contains a different asset revision; choose a separate directory")
    info = HfApi().model_info(REPOSITORY, revision=revision, files_metadata=True, token=False)
    if info.sha != revision:
        raise ValueError("Hub metadata did not resolve to the requested immutable revision")
    filenames = select_assets(info.siblings)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for filename in filenames:
        # Exact filename + immutable revision only; no snapshot/pattern download can pull model shards.
        if not is_allowed_asset(filename):
            raise ValueError(f"Forbidden model payload: {filename}")
        local = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                revision=revision,
                local_dir=str(destination),
                token=False,
            )
        )
        if local.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Downloaded asset exceeds limit: {filename}")
        files.append({"file": filename, "size_bytes": local.stat().st_size, "sha256": file_sha256(local)})
    manifest = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": revision,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "weights_downloaded_by_this_script": False,
        "files": files,
        "tokenizer_check": verify_tokenizer_assets(destination),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "models/Qwen3.5-9B")
    parser.add_argument("--revision", default=PINNED_REVISION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect-revision", action="store_true", help="Read metadata only; download no files")
    mode.add_argument(
        "--verify-local", action="store_true", help="Verify prepared assets without network access"
    )
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.inspect_revision:
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            REPOSITORY, revision=args.revision or "main", files_metadata=True, token=False
        )
        print(
            json.dumps(
                {
                    "repository": REPOSITORY,
                    "revision": info.sha,
                    "allowlisted_files": select_assets(info.siblings),
                },
                indent=2,
            )
        )
        return
    if not args.revision:
        parser.error("Set an immutable --revision after metadata inspection")
    if args.verify_local:
        print(json.dumps(verify_local_assets(args.destination, args.revision), ensure_ascii=False, indent=2))
        return
    print(json.dumps(prepare_assets(args.destination, args.revision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
