"""Thread-safe call reservations and honest accounting of missing usage."""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from repro.artifacts.serialization import utc_now


class BudgetExceeded(RuntimeError):
    """A request cannot be admitted within the configured run budget."""


@dataclass(frozen=True)
class _Reservation:
    role: str
    max_tokens: int
    metadata: dict[str, Any]
    timestamp: str


class UsageLedger:
    """Reserve output limits before calls and reconcile known actual usage.

    Failed calls are charged a call. When the backend omits usage (including a
    timeout after generation), their full output reservation remains charged and
    the summary explicitly reports incomplete accounting. This is a conservative
    budget ledger, not a claim that all reserved tokens were generated.
    """

    def __init__(
        self,
        max_calls: int | None = None,
        max_completion_tokens: int | None = None,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        for name, value in (("max_calls", max_calls), ("max_completion_tokens", max_completion_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        self.max_calls = max_calls
        self.max_completion_tokens = max_completion_tokens
        self._sink = sink
        self._lock = threading.RLock()
        self._pending: dict[str, _Reservation] = {}
        self._calls = 0
        self._charged = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._finished = 0
        self._unknown = 0
        self._failed = 0
        self._latency = 0.0
        self._by_role: dict[str, dict[str, int]] = {}

    def begin_call(self, role: str, max_tokens: int, metadata: dict[str, Any] | None = None) -> str:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        with self._lock:
            if self.max_calls is not None and self._calls + 1 > self.max_calls:
                raise BudgetExceeded(f"Model-call budget exceeded ({self.max_calls})")
            if (
                self.max_completion_tokens is not None
                and self._charged + max_tokens > self.max_completion_tokens
            ):
                raise BudgetExceeded(
                    f"Output-token budget cannot reserve {max_tokens} tokens "
                    f"({self._charged}/{self.max_completion_tokens} already charged or reserved)"
                )
            call_id = uuid.uuid4().hex
            reservation = _Reservation(str(role), max_tokens, dict(metadata or {}), utc_now())
            self._pending[call_id] = reservation
            self._calls += 1
            self._charged += max_tokens
            role_counts = self._by_role.setdefault(
                reservation.role,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "unknown_usage_calls": 0,
                    "failed_calls": 0,
                    "pending_calls": 0,
                    "charged_completion_tokens": 0,
                },
            )
            role_counts["calls"] += 1
            role_counts["pending_calls"] += 1
            role_counts["charged_completion_tokens"] += max_tokens
            if self._sink:
                self._sink(
                    {
                        "event": "call_started",
                        "call_id": call_id,
                        "role": role,
                        "reserved_completion_tokens": max_tokens,
                        "metadata": reservation.metadata,
                        "timestamp": reservation.timestamp,
                    }
                )
            return call_id

    def finish_call(
        self,
        call_id: str,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_s: float,
        status: str = "ok",
        error: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        for name, count in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens)):
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if not math.isfinite(latency_s) or latency_s < 0:
            raise ValueError("latency_s must be a finite nonnegative value")
        with self._lock:
            if call_id not in self._pending:
                raise ValueError("Call is unknown or has already been finalized")
            reservation = self._pending.pop(call_id)
            role_counts = self._by_role[reservation.role]
            role_counts["pending_calls"] -= 1
            complete_usage = prompt_tokens is not None and completion_tokens is not None
            if completion_tokens is not None:
                self._charged += completion_tokens - reservation.max_tokens
                self._completion_tokens += completion_tokens
                role_counts["completion_tokens"] += completion_tokens
                role_counts["charged_completion_tokens"] += completion_tokens - reservation.max_tokens
            if prompt_tokens is not None:
                self._prompt_tokens += prompt_tokens
                role_counts["prompt_tokens"] += prompt_tokens
            self._finished += 1
            self._failed += status != "ok"
            self._unknown += not complete_usage
            role_counts["failed_calls"] += status != "ok"
            role_counts["unknown_usage_calls"] += not complete_usage
            self._latency += latency_s
            record = {
                **extra,
                "event": "call_finished",
                "call_id": call_id,
                "role": reservation.role,
                "metadata": reservation.metadata,
                "started_at": reservation.timestamp,
                "timestamp": utc_now(),
                "status": status,
                "error": error,
                "latency_s": latency_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usage_complete": complete_usage,
                "reserved_completion_tokens": reservation.max_tokens,
                "charged_completion_tokens": (
                    completion_tokens if completion_tokens is not None else reservation.max_tokens
                ),
            }
            if self._sink:
                self._sink(record)
            return record

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self._calls,
                "completed_calls": self._finished,
                "failed_calls": self._failed,
                "pending_calls": len(self._pending),
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "charged_completion_tokens": self._charged,
                "unknown_usage_calls": self._unknown,
                "usage_complete": not self._unknown and not self._pending,
                "sum_call_latency_s": self._latency,
                "by_role": {role: dict(counts) for role, counts in self._by_role.items()},
                "limits": {
                    "max_calls": self.max_calls,
                    "max_completion_tokens": self.max_completion_tokens,
                },
            }
