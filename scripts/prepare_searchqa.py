"""Prepare fixed SearchQA IDs from local data or the official public HF source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro.config import load_config, resolve_path
from repro.datasets.searchqa import prepare_searchqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Shared preparation settings")
    parser.add_argument(
        "--raw", type=Path, help="Local JSON/JSONL file or directory; omit to download selected rows"
    )
    parser.add_argument(
        "--manifest-dir", type=Path
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train", type=int)
    parser.add_argument("--val", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download-mode", choices=("parquet", "viewer", "streaming"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", help="Fixed HF dataset revision for parquet or streaming mode")
    parser.add_argument("--no-course-limits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, presets=["course", "searchqa.course"])
    settings = config["preparation"]
    output = args.output if args.output is not None else resolve_path(config["data"]["split_dir"])
    manifest_dir = args.manifest_dir or resolve_path(settings["searchqa_manifest_dir"])
    cache_dir = args.cache_dir or resolve_path(settings["raw_cache_dir"])
    raw = args.raw or (resolve_path(settings["searchqa_raw_path"]) if settings["searchqa_raw_path"] else None)
    download_mode = args.download_mode or settings["searchqa_download_mode"]
    revision = args.revision
    if revision is None and download_mode != "viewer":
        revision = settings["searchqa_revision"]
    sizes = {
        split: getattr(args, split) if getattr(args, split) is not None else settings[f"{split}_count"]
        for split in ("train", "val", "test")
    }
    manifest = prepare_searchqa(
        raw,
        manifest_dir,
        output,
        sizes,
        seed=args.seed if args.seed is not None else settings["seed"],
        hf_revision=revision,
        download_mode=download_mode,
        cache_dir=cache_dir,
        course_limits=not args.no_course_limits,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": {split: info["count"] for split, info in manifest["splits"].items()},
                "source": manifest["source"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
