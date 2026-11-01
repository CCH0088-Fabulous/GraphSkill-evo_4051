"""Stable optimizer sampling seeds, independent of Python's randomized hash."""

import hashlib
import json


def optimizer_seed(base_seed: int, phase: str, stage: str, attempt: int) -> int:
    payload = json.dumps(["optimizer_attempt_v1", base_seed, phase, stage, attempt], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def optimizer_retry_seed(base_seed: int, phase: str, stage: str, attempt: int) -> int:
    if attempt == 1:
        return base_seed
    payload = json.dumps(["optimizer_retry_v1", base_seed, phase, stage, attempt], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")
