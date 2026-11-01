"""Explicit synthetic backend for plumbing checks; never a research result."""

from __future__ import annotations

import json
import re
from typing import Any

from repro.contracts import Generation
from repro.inference.accounting import UsageLedger

SYNTHETIC_SKILL = """# Synthetic infrastructure skill

## Global Guidance
This synthetic document tests infrastructure only. Ground answers in the provided context.

## Node Lists
### Read evidence
Read the question and supplied context.
### Return answer
Return a short answer inside <answer> tags.

## Task Graphs
### Question answering
**Use when:** A text question and context are supplied.
**Workflow:**
1. Read evidence
2. Return answer
"""


class OfflineClient:
    """Return fixed protocol fixtures with visibly synthetic token estimates."""

    def __init__(self, ledger: UsageLedger) -> None:
        self.ledger = ledger
        self.seed = 42

    def close(self) -> None:
        pass

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "target",
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Generation:
        text = "\n".join(message["content"] for message in messages)
        if role == "optimizer":
            match = re.search(r"Number of skills to generate:\s*(\d+)", text)
            content = (
                json.dumps({"skills": [SYNTHETIC_SKILL] * int(match[1])})
                if match
                else json.dumps(
                    {
                        "new_skill": SYNTHETIC_SKILL,
                        "notes": ["Synthetic infrastructure check only"],
                    }
                )
            )
        else:
            content = "<answer>synthetic-unanswered</answer>"
        prompt_tokens = len(text.split())
        completion_tokens = len(content.split())
        call_id = self.ledger.begin_call(
            role,
            max_tokens or 2048,
            {
                **(metadata or {}),
                "synthetic": True,
                "token_count_method": "whitespace_estimate",
            },
        )
        self.ledger.finish_call(
            call_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=0.0,
            synthetic=True,
            request={"messages": messages, "seed": self.seed if seed is None else seed},
            response={"content": content},
        )
        return Generation(
            content,
            prompt_tokens,
            completion_tokens,
            0.0,
            finish_reason="synthetic",
            raw_response={"synthetic": True},
        )
