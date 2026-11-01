"""The inspected upstream policy: seeded random sampling of EM=0 failures."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


class OriginalFailureSelector:
    def select(self, rows: Sequence[dict[str, Any]], *, k: int, seed: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        failures = []
        for row in rows:
            if "em" not in row and "hard" not in row:
                raise ValueError("Failure selection needs the em or upstream hard score")
            score = float(row["em"] if "em" in row else row["hard"])
            if score < 1e-9:
                failures.append(row)
        return random.Random(seed).sample(failures, k=min(k, len(failures)))
