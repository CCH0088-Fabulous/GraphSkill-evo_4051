"""Resumable, revision-pinned downloads of public dataset Parquet files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from repro.datasets.manifests import file_sha256

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def get_json(client: httpx.Client, url: str, *, attempts: int = 6, **params: Any) -> Any:
    """Retry transient transport/server errors without changing requested identities."""
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in RETRY_STATUS:
                raise
            if attempt + 1 == attempts:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print(f"Dataset metadata retry {attempt + 1}/{attempts} in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Unreachable download retry state")


def download_file(
    client: httpx.Client,
    url: str,
    target: Path,
    expected_size: int,
    *,
    attempts: int = 6,
) -> dict[str, Any]:
    """Retain completed files; resume .part files and rename only after size validation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(target.suffix + ".download.json")
    if target.exists():
        if target.stat().st_size != expected_size:
            raise ValueError(f"Completed file has an unexpected size: {target}")
        checksum = file_sha256(target)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("sha256") != checksum:
                raise ValueError(f"Cached dataset checksum mismatch: {target}")
            return receipt
        receipt = {"url": url, "bytes": expected_size, "sha256": checksum}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                raise ValueError(f"Partial dataset exceeds expected size: {partial}")
            if offset != expected_size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                print(f"Downloading {target.name}: {offset}/{expected_size} bytes", flush=True)
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    if response.status_code == 206:
                        content_range = response.headers.get("content-range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError(f"Unexpected range response for {target.name}")
                    downloaded = offset if mode == "ab" else 0
                    report_at = downloaded + 32 * 1024 * 1024
                    with partial.open(mode) as stream:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            stream.write(chunk)
                            downloaded += len(chunk)
                            if downloaded >= report_at:
                                print(f"  {target.name}: {downloaded}/{expected_size}", flush=True)
                                report_at = downloaded + 32 * 1024 * 1024
            if partial.stat().st_size != expected_size:
                raise httpx.ReadError(
                    f"Incomplete dataset transfer: {partial.stat().st_size}/{expected_size}"
                )
            checksum = file_sha256(partial)
            partial.replace(target)
            receipt = {"url": url, "bytes": expected_size, "sha256": checksum}
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            return receipt
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in RETRY_STATUS:
                raise
            if attempt + 1 == attempts:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print(f"Dataset transfer retry {attempt + 1}/{attempts} in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Unreachable download retry state")


def download_hf_parquet(
    dataset: str,
    cache_dir: str | Path,
    *,
    prefix: str,
    revision: str | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Download only the named dataset configuration, never model repositories."""
    root = Path(cache_dir).resolve()
    with httpx.Client(timeout=httpx.Timeout(120, connect=30), follow_redirects=True) as client:
        if revision is None:
            info = get_json(client, f"https://huggingface.co/api/datasets/{dataset}")
            revision = info.get("sha")
        if not isinstance(revision, str) or not revision:
            raise ValueError("A fixed dataset revision is required")
        tree = get_json(
            client,
            f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}",
            recursive="true",
        )
        files = [
            item
            for item in tree
            if item.get("type") == "file"
            and item["path"].startswith(prefix)
            and item["path"].endswith(".parquet")
        ]
        if not files:
            raise ValueError(f"No Parquet data found for {dataset} prefix {prefix!r}")
        paths = []
        receipts = {}
        for item in sorted(files, key=lambda entry: entry["path"]):
            relative = item["path"]
            target = (root / dataset.replace("/", "--") / revision / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Dataset path escapes its cache directory")
            url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{relative}?download=true"
            receipts[relative] = download_file(client, url, target, item["size"])
            paths.append(target)
    return paths, {
        "retrieval_backend": "revision_pinned_hf_parquet",
        "dataset": dataset,
        "revision": revision,
        "config_prefix": prefix,
        "raw_files": receipts,
    }
