"""The original failure selector remains comparable to the inspected baseline."""

import random

import pytest

from repro.selectors import OriginalFailureSelector, get_selector


def test_original_policy_matches_seeded_upstream_sampling():
    rows = [{"id": str(i), "hard": int(i % 3 == 0)} for i in range(20)]
    expected = random.Random(42).sample([row for row in rows if row["hard"] < 1e-9], 5)
    assert OriginalFailureSelector().select(rows, k=5, seed=42) == expected
    assert len(OriginalFailureSelector().select(rows, k=100, seed=42)) == 13
    assert get_selector("original").select(rows, k=0, seed=42) == []


def test_failure_policy_requires_scores_and_does_not_fake_innovation():
    with pytest.raises(ValueError, match="score"):
        OriginalFailureSelector().select([{"id": "a"}], k=1, seed=42)
    for name in ("improved", "ablation"):
        with pytest.raises(NotImplementedError, match="member B"):
            get_selector(name)
