"""Prepare course-provided files, or a disclosed custom official-public HotpotQA subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro.config import load_config, resolve_path
from repro.datasets.hotpotqa import prepare_hotpotqa, prepare_public_hotpotqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Shared preparation settings")
    parser.add_argument("--source", choices=("official-public", "course"))
    parser.add_argument("--raw-train", type=Path)
    parser.add_argument(
        "--raw-val", type=Path, help="Course validation file, or official dev file in public mode"
    )
    parser.add_argument("--raw-test", type=Path, help="Course-provided labeled test file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train", type=int)
    parser.add_argument("--val", type=int)
    parser.add_argument("--test", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--download-mode", choices=("parquet", "viewer"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", help="Fixed official dataset revision in parquet mode")
    parser.add_argument("--no-course-limits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, presets=["course", "hotpotqa.course"])
    settings = config["preparation"]
    output = args.output if args.output is not None else resolve_path(config["data"]["split_dir"])
    cache_dir = args.cache_dir or resolve_path(settings["raw_cache_dir"])
    source = args.source or settings["hotpotqa_source"]
    download_mode = args.download_mode or settings["hotpotqa_download_mode"]
    raw_paths = {
        split: getattr(args, f"raw_{split}")
        or (resolve_path(settings[f"hotpotqa_raw_{split}"]) if settings[f"hotpotqa_raw_{split}"] else None)
        for split in ("train", "val", "test")
    }
    revision = args.revision
    if revision is None and download_mode != "viewer":
        revision = settings["hotpotqa_revision"]
    sizes = {
        split: getattr(args, split) if getattr(args, split) is not None else settings[f"{split}_count"]
        for split in ("train", "val", "test")
    }
    options = {
        "seed": args.seed if args.seed is not None else settings["seed"],
        "course_limits": not args.no_course_limits,
        "overwrite": args.overwrite,
    }
    if source == "course":
        paths = {split: path for split, path in raw_paths.items() if path is not None}
        manifest = prepare_hotpotqa(paths, output, sizes, **options)
    else:
        if raw_paths["test"] is not None:
            parser.error("Public mode uses labeled official dev as local test; --raw-test is for course mode")
        manifest = prepare_public_hotpotqa(
            output,
            sizes,
            raw_train_path=raw_paths["train"],
            raw_dev_path=raw_paths["val"],
            download_mode=download_mode,
            cache_dir=cache_dir,
            hf_revision=revision,
            **options,
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
