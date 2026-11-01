"""Shared records only; no data access, model calls, or scoring side effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Split = Literal["train", "val", "test"]
Condition = Literal["no_skill", "initial_skill", "original", "improved", "ablation"]


@dataclass(frozen=True)
class QAExample:
    id: str
    dataset: str
    split: Split
    question: str
    context: str
    answers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Generation:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    finish_reason: str | None = None
    response_id: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    dataset: str
    split: Split
    condition: str
    skill_id: str
    raw_output: str
    answer: str
    em: float
    f1: float
    latency_s: float
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
