"""Small public-dataset downloads through the documented Hugging Face Viewer API."""

from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from repro.datasets.downloads import get_json

VIEWER = "https://datasets-server.huggingface.co"


def _response_json(
    client: httpx.Client,
    endpoint: str,
    *,
    cache_revision: str | None = None,
    **params: Any,
) -> dict[str, Any]:
    cache_path = None
    if cache_revision is not None:
        identity = json.dumps(
            {"endpoint": endpoint, "revision": cache_revision, "params": params}, sort_keys=True
        ).encode()
        cache_path = Path("data/cache/viewer") / f"{hashlib.sha256(identity).hexdigest()}.json"
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = get_json(client, f"{VIEWER}/{endpoint}", **params)
    if not isinstance(payload, dict):
        raise ValueError("Dataset Viewer returned an unexpected response")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def dataset_revision(client: httpx.Client, dataset: str) -> str:
    revision = get_json(client, f"https://huggingface.co/api/datasets/{dataset}").get("sha")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Could not resolve dataset revision for {dataset}")
    return revision


def _unpack_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("partial"):
        raise ValueError("Dataset Viewer returned a partial query; use revision-pinned streaming instead")
    records = []
    for item in payload.get("rows", []):
        if item.get("truncated_cells"):
            raise ValueError("Dataset Viewer truncated record fields; use streaming instead")
        row = item.get("row")
        if not isinstance(row, dict):
            raise ValueError("Dataset Viewer row is not an object")
        records.append(row)
    return records


def viewer_records_by_ids(
    dataset: str,
    identities: list[str],
    *,
    id_column: str = "key",
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve only requested IDs; save exact payload hashes for reproducibility.

    The Viewer API exposes the current dataset, not a revision-pinned historical API.
    We record its limitation explicitly and reject dataset changes during retrieval.
    """
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        revision = dataset_revision(client, dataset)
        splits = _response_json(client, "splits", dataset=dataset).get("splits", [])
        if not splits:
            raise ValueError(f"No Viewer splits found for {dataset}")
        batches = [identities[index : index + 5] for index in range(0, len(identities), 5)]
        jobs = [(split, batch) for split in splits for batch in batches]

        def fetch(job: tuple[dict[str, Any], list[str]]) -> list[dict[str, Any]]:
            split, batch = job
            predicates = [f"\"{id_column}\"='{identity.replace(chr(39), chr(39) * 2)}'" for identity in batch]
            payload = _response_json(
                client,
                "filter",
                cache_revision=revision,
                dataset=dataset,
                config=split["config"],
                split=split["split"],
                where=" OR ".join(predicates),
                offset=0,
                length=100,
            )
            if payload.get("num_rows_total", 0) > 100:
                raise ValueError("Requested IDs produced too many rows")
            return _unpack_rows(payload)

        records = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(fetch, jobs):
                records.extend(batch)
        if dataset_revision(client, dataset) != revision:
            raise RuntimeError("Dataset revision changed during retrieval; retry into a new output directory")
    metadata = {
        "retrieval_backend": "huggingface_dataset_viewer_filter",
        "revision_at_retrieval": revision,
        "revision_pinned_api": False,
        "revision_check": "hub_revision_unchanged_before_after",
        "records_sha256": hashlib.sha256(
            json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    return records, metadata


def viewer_random_rows(
    dataset: str,
    config: str,
    split: str,
    count: int,
    *,
    seed: int,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fix row offsets before inspecting answers and fetch only those records."""
    if count < 0:
        raise ValueError("Row count must be non-negative")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        revision = dataset_revision(client, dataset)
        preview = _response_json(
            client,
            "rows",
            cache_revision=revision,
            dataset=dataset,
            config=config,
            split=split,
            offset=0,
            length=1,
        )
        total = preview.get("num_rows_total")
        if not isinstance(total, int) or count > total:
            raise ValueError(f"Requested {count} rows, dataset reports {total}")
        offsets = sorted(random.Random(seed).sample(range(total), count))

        def fetch(offset: int) -> dict[str, Any]:
            payload = _response_json(
                client,
                "rows",
                cache_revision=revision,
                dataset=dataset,
                config=config,
                split=split,
                offset=offset,
                length=1,
            )
            records = _unpack_rows(payload)
            if len(records) != 1:
                raise ValueError(f"Missing Viewer row at offset {offset}")
            return records[0]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(fetch, offsets))
        if dataset_revision(client, dataset) != revision:
            raise RuntimeError("Dataset revision changed during retrieval")
    return records, {
        "retrieval_backend": "huggingface_dataset_viewer_rows",
        "revision_at_retrieval": revision,
        "revision_pinned_api": False,
        "revision_check": "hub_revision_unchanged_before_after",
        "source_split": split,
        "source_rows": total,
        "selected_row_offsets": offsets,
        "row_offset_selection": "random.Random(seed).sample before row retrieval",
        "seed": seed,
    }
