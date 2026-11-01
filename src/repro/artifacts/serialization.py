"""JSON conversion and secret redaction shared by artifact writers."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SECRET_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "token", "password",
    "authorization", "client_secret", "secret", "credentials",
}
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def redact(value: Any) -> Any:
    """Convert common records to JSON values and remove recognized credentials.

    Token *counts* are kept: keys such as ``prompt_tokens`` and
    ``max_completion_tokens`` are accounting, not authentication credentials.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            secret = normalized in _SECRET_KEYS or normalized.endswith(
                ("_api_key", "_password", "_secret", "_token")
            )
            result[str(key)] = "[REDACTED]" if secret else redact(item)
        return result
    if isinstance(value, (tuple, list, set, frozenset)):
        return [redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return redact(value.value)
    if isinstance(value, str):
        return _API_KEY_RE.sub("[REDACTED]", _BEARER_RE.sub("Bearer [REDACTED]", value))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def json_text(value: Any, *, pretty: bool = False) -> str:
    """Encode strict JSON; reject NaN rather than silently corrupt result files."""
    return json.dumps(redact(value), ensure_ascii=False, allow_nan=False, indent=2 if pretty else None)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()
