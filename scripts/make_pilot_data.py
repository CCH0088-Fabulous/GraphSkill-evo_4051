"""Create small pilots exclusively from previously frozen training/development IDs."""

import argparse

from repro.config import load_config, resolve_path
from repro.datasets.io import load_split
from repro.datasets.manifests import file_sha256, freeze_splits, verify_manifest, verify_manifest_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Shared preparation settings")
    args = parser.parse_args()
    for dataset in ("searchqa", "hotpotqa"):
        course = load_config(args.config, presets=["course", f"{dataset}.course"])
        pilot = load_config(args.config, presets=[f"{dataset}.pilot"])
        settings = course["preparation"]
        parent = resolve_path(course["data"]["split_dir"])
        source_manifest = verify_manifest(parent)
        if source_manifest["dataset"] != dataset:
            raise ValueError(f"{parent}: source dataset mismatch")
        if course["data"]["manifest"]:
            verify_manifest_reference(source_manifest, resolve_path(course["data"]["manifest"]))
        output = resolve_path(pilot["data"]["split_dir"])
        sizes = {"train": settings["pilot_train_count"], "val": settings["pilot_val_count"], "test": 0}
        source_hash = file_sha256(parent / "split_manifest.json")
        if (output / "split_manifest.json").exists():
            existing = verify_manifest(output)
            if existing["source"].get("parent_manifest_sha256") != source_hash:
                raise ValueError(f"{output}: parent dataset changed; use a new pilot directory")
            counts = {split: existing["splits"][split]["count"] for split in sizes}
            if counts != sizes or existing["seed"] != settings["pilot_seed"]:
                raise ValueError(f"{output}: pilot settings differ; use a new pilot directory")
            print(f"{dataset}: existing pilot verified")
            continue
        freeze_splits(
            dataset,
            {split: load_split(parent, split, dataset) for split in ("train", "val")},
            output,
            sizes,
            seed=settings["pilot_seed"],
            source={
                "parent_manifest_sha256": source_hash,
                "parent_dir": course["data"]["split_dir"],
                "parent_source": source_manifest["source"],
                "note": "No held-out test examples in pilot",
            },
        )
        print(f"{dataset}: pilot train={sizes['train']}, val={sizes['val']}, test=0")


if __name__ == "__main__":
    main()
