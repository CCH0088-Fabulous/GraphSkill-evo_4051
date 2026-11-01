"""Budget tests exercise concurrency, retries, and missing backend usage."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from repro.inference.accounting import BudgetExceeded, UsageLedger


def test_reservations_prevent_parallel_budget_oversubscription():
    ledger = UsageLedger(max_calls=20, max_completion_tokens=50)

    def attempt(_):
        try:
            return ledger.begin_call("target", 20)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(20)))
    accepted = [value for value in results if value]
    assert len(accepted) == 2
    assert ledger.summary()["calls"] == 2
    assert ledger.summary()["charged_completion_tokens"] == 40


def test_reconcile_known_usage_releases_unused_reservation():
    ledger = UsageLedger(max_calls=2, max_completion_tokens=30)
    call_id = ledger.begin_call("optimizer", 30)
    ledger.finish_call(call_id, prompt_tokens=10, completion_tokens=5, latency_s=1)
    second_id = ledger.begin_call("target", 25)
    ledger.finish_call(second_id, prompt_tokens=20, completion_tokens=15, latency_s=2)
    summary = ledger.summary()
    assert summary["calls"] == 2
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 20
    assert summary["charged_completion_tokens"] == 20
    assert summary["by_role"]["optimizer"]["completion_tokens"] == 5
    assert summary["by_role"]["target"]["completion_tokens"] == 15
    assert summary["usage_complete"]
    with pytest.raises(BudgetExceeded, match="call"):
        ledger.begin_call("target", 1)


def test_timeout_keeps_unknown_usage_and_retry_is_a_new_call():
    records = []
    ledger = UsageLedger(max_calls=2, max_completion_tokens=50, sink=records.append)
    failed_id = ledger.begin_call("target", 20, {"attempt": 1})
    failed = ledger.finish_call(
        failed_id, prompt_tokens=None, completion_tokens=None,
        latency_s=5, status="error", error="timeout",
    )
    retry_id = ledger.begin_call("target", 20, {"attempt": 2})
    ledger.finish_call(retry_id, prompt_tokens=7, completion_tokens=4, latency_s=1)
    summary = ledger.summary()
    assert summary["calls"] == 2
    assert summary["failed_calls"] == 1
    assert summary["unknown_usage_calls"] == 1
    assert not summary["usage_complete"]
    assert summary["completion_tokens"] == 4
    assert summary["charged_completion_tokens"] == 24
    assert summary["by_role"]["target"]["calls"] == 2
    assert summary["by_role"]["target"]["unknown_usage_calls"] == 1
    assert failed["completion_tokens"] is None
    assert [row["event"] for row in records] == [
        "call_started", "call_finished", "call_started", "call_finished",
    ]
    with pytest.raises(ValueError, match="finalized"):
        ledger.finish_call(failed_id, prompt_tokens=0, completion_tokens=0, latency_s=0)


def test_pending_calls_are_not_reported_as_complete_accounting():
    ledger = UsageLedger()
    ledger.begin_call("target", 10)
    assert not ledger.summary()["usage_complete"]
    assert ledger.summary()["pending_calls"] == 1


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1])
def test_nonfinite_call_latency_is_rejected_before_finalizing(latency):
    ledger = UsageLedger()
    call_id = ledger.begin_call("target", 10)
    with pytest.raises(ValueError, match="finite nonnegative"):
        ledger.finish_call(call_id, prompt_tokens=1, completion_tokens=1, latency_s=latency)
    assert ledger.summary()["pending_calls"] == 1
