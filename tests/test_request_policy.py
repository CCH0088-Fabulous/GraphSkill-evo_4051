"""Truncation and structural retries are explicit, metered and reproducible."""

import json
from copy import deepcopy

import httpx
import pytest

from repro.artifacts import RunStore
from repro.config import DEFAULTS
from repro.contracts import QAExample
from repro.inference.accounting import UsageLedger
from repro.inference.client import ChatClient
from repro.inference.seeding import optimizer_retry_seed
from repro.integration.model_bridge import OperatorAttemptLimit, OptimizerBridge
from repro.integration.qa_adapter import execute_examples


def client(store, respond):
    cfg = deepcopy(DEFAULTS["model"])
    cfg.update(temperature=0.6, top_p=1.0, top_k=50)
    return ChatClient(
        cfg,
        UsageLedger(
            max_calls=10, max_completion_tokens=10000, sink=lambda row: store.append_jsonl("calls.jsonl", row)
        ),
        transport=httpx.MockTransport(respond),
    )


def answer(reason, text="<answer>Paris</answer>"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        },
    )


def test_network_retry_preserves_payload_seed_and_logical_attempt(tmp_path):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(503) if len(seen) == 1 else answer("stop")

    store = RunStore(tmp_path)
    chat = client(store, respond)
    chat.generate([{"role": "user", "content": "question"}], seed=123, metadata={"operator_attempt": 7})
    chat.close()
    assert seen[0] == seen[1] and seen[0]["seed"] == 123
    assert (seen[0]["top_p"], seen[0]["top_k"]) == (1.0, 50)
    records = [json.loads(line) for line in (store.path / "calls.jsonl").read_text().splitlines()]
    starts = [r for r in records if r["event"] == "call_started"]
    assert [r["metadata"]["http_attempt"] for r in starts] == [1, 2]
    assert all(r["metadata"]["operator_attempt"] == 7 for r in starts)


def test_optimizer_rejects_truncation_and_advances_reproducible_seed(tmp_path):
    sequences = []
    for repetition in range(2):
        seen = []

        def respond(request, seen=seen):
            seen.append(json.loads(request.content)["seed"])
            return answer("length" if len(seen) == 1 else "stop", text='{"new_skill":"fixture"}')

        store = RunStore(tmp_path / str(repetition))
        chat = client(store, respond)
        bridge = OptimizerBridge(chat, store, seed_policy="optimizer_retry_v1")
        with bridge.phase("initialization", 2):
            text, _usage = bridge.chat_optimizer("system", "user", stage="init", max_completion_tokens=30)
        chat.close()
        assert text == '{"new_skill":"fixture"}'
        assert seen == [optimizer_retry_seed(42, "initialization", "init", n) for n in (1, 2)]
        assert seen[0] == 42
        assert len(set(seen)) == 2
        assert json.loads((store.path / "operators/call_00001.json").read_text())["output_truncated"]
        sequences.append(seen)
    assert sequences[0] == sequences[1]


def test_optimizer_truncation_cannot_bypass_attempt_bound(tmp_path):
    store = RunStore(tmp_path)
    chat = client(store, lambda request: answer("length"))
    bridge = OptimizerBridge(chat, store, seed_policy="optimizer_retry_v1")
    with bridge.phase("init", 2), pytest.raises(OperatorAttemptLimit):
        bridge.chat_optimizer("system", "user", max_completion_tokens=30)
    assert chat.ledger.summary()["calls"] == 2
    assert chat.ledger.summary()["unknown_usage_calls"] == 0
    chat.close()


def test_truncated_qa_is_zero_even_if_partial_output_contains_correct_answer(tmp_path):
    store = RunStore(tmp_path)
    chat = client(store, lambda request: answer("length"))
    example = QAExample("example", "searchqa", "train", "Which city?", "Paris is a city.", ("Paris",))
    row = execute_examples([example], "", chat, store, "original")[0]
    chat.close()
    assert row["status"] == "truncated" and row["output_truncated"]
    assert row["em"] == row["f1"] == 0 and row["predicted_answer"] == ""
    assert row["raw_output"] == "<answer>Paris</answer>"
    assert row["failure_type"] == "output_truncated" and "truncated" in row["fail_reason"]


def test_each_optimizer_stage_starts_with_base_seed_then_retries_change_seed(tmp_path):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content)["seed"])
        return answer("stop")

    store = RunStore(tmp_path)
    chat = client(store, respond)
    bridge = OptimizerBridge(chat, store, seed_policy="optimizer_retry_v1")
    with bridge.phase("generation_1", 4):
        for stage in ("mutation_other", "mutation_graph", "mutation_graph"):
            bridge.chat_optimizer("system", "user", stage=stage, max_completion_tokens=30)
    chat.close()
    assert seen[:2] == [42, 42]
    assert seen[2] == optimizer_retry_seed(42, "generation_1", "mutation_graph", 2)
    assert seen[2] != 42
