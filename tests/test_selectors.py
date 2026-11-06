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


# ---- placeholder selectors for verifying the improved/ablation switch (not B's method) ----

class _FirstKFailures:
    """Deterministic stand-in for B's selector: the first k EM=0 rows. Records every call."""

    def __init__(self):
        self.calls = []

    def select(self, rows, *, k, seed):
        self.calls.append({"n_rows": len(rows), "k": k, "seed": seed})
        return [row for row in rows if float(row.get("em", row.get("hard", 1))) < 1e-9][:k]


@pytest.fixture
def registered(monkeypatch):
    """Register placeholder selectors under B's reserved names; always restore the registry."""
    from repro import selectors

    monkeypatch.setattr(selectors, "_REGISTRY", dict(selectors._REGISTRY))
    placeholders = {name: _FirstKFailures() for name in ("improved", "ablation")}
    for name, selector in placeholders.items():
        selectors.register_selector(name, selector)
    return placeholders


def test_registered_names_replace_the_reserved_error_and_original_is_untouched(registered):
    rows = [{"id": str(i), "hard": int(i % 2 == 0)} for i in range(10)]
    for name, selector in registered.items():
        assert get_selector(name) is selector
        assert [row["id"] for row in get_selector(name).select(rows, k=2, seed=1)] == ["1", "3"]
    assert isinstance(get_selector("original"), OriginalFailureSelector)


def test_registry_is_restored_between_tests():
    for name in ("improved", "ablation"):
        with pytest.raises(NotImplementedError, match="member B"):
            get_selector(name)


def test_registering_twice_or_with_a_bad_name_is_rejected(registered):
    from repro.selectors import register_selector

    with pytest.raises(ValueError, match="already registered"):
        register_selector("improved", _FirstKFailures())
    with pytest.raises(ValueError, match="lowercase"):
        register_selector("Bad-Name", _FirstKFailures())
    with pytest.raises(TypeError, match="select"):
        register_selector("no_select", object())
