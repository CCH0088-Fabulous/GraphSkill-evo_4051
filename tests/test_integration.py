"""Exercise the actual upstream loop without downloading any model weights."""

import json
from copy import deepcopy

import httpx
import pytest

from repro.artifacts import RunStore
from repro.config import DEFAULTS, PROJECT_ROOT, load_config
from repro.contracts import QAExample
from repro.inference.accounting import BudgetExceeded, UsageLedger
from repro.inference.client import ChatClient
from repro.inference.offline import OfflineClient
from repro.integration import run_evolution
from repro.integration.dataloader import QADataLoader
from repro.integration.model_bridge import OperatorAttemptLimit
from repro.integration.qa_adapter import QAAdapter, execute_examples
from repro.integration.source import ensure_upstream


def fixture_config(dataset="searchqa"):
    config = load_config(PROJECT_ROOT / "configs" / "offline.searchqa.yaml")
    config["data"]["dataset"] = dataset
    config["data"]["split_dir"] = str(PROJECT_ROOT / "tests" / "fixtures" / dataset)
    return config


def example(split="val", identifier="sample-1"):
    return QAExample(identifier, "searchqa", split, "Which city?", "The city is Paris.", ("Paris",))


def offline_client(store, *, max_calls=100):
    return OfflineClient(UsageLedger(
        max_calls=max_calls, max_completion_tokens=100000,
        sink=lambda row: store.append_jsonl("calls.jsonl", row),
    ))


@pytest.mark.parametrize("dataset", ["searchqa", "hotpotqa"])
def test_actual_upstream_loop_exports_bounded_candidates_and_full_artifacts(tmp_path, dataset):
    config = fixture_config(dataset)
    config["evolution"]["generations"] = 2
    store = RunStore(tmp_path, dataset)
    client = offline_client(store)
    result = run_evolution(config, client, store)
    history = json.loads((store.path / "upstream/history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert [row["operator"] for row in history[0]["operator_records"]] == [
        "mutation_other", "mutation_graph",
    ]
    assert [row["operator"] for row in history[1]["operator_records"]] == [
        "crossover_other", "crossover_graph",
    ]
    assert all(len(row["offspring"]) == 2 for row in history)
    assert history[0]["operator_records"][0]["tokens"]["optimizer"]["calls"] == 1
    assert (store.path / "upstream/best_skill.md").is_file()
    assert (store.path / "graphs/lineage.mmd").is_file()
    assert (store.path / "graphs/g0001_i0000.mmd").is_file()
    predictions = [
        json.loads(row)
        for row in (store.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["split"] for row in predictions} == {"train", "val"}
    assert all("evaluation_id" in row for row in predictions)
    assert result["tokens"]["_total"]["calls"] == client.ledger.summary()["calls"]
    assert result["accounting"]["usage_complete"]
    first_record = history[0]["operator_records"][0]
    assert "[EVALUATION RESULT]" in first_record["prompt"]
    assert "#### Hidden Reference" not in first_record["prompt"]
    # Every initialized and newly generated candidate is archived, even if it is not retained.
    assert len(list((store.path / "skills").glob("*.md"))) == 6


def test_repeated_training_evaluations_remain_distinct_and_test_is_explicit(tmp_path):
    store = RunStore(tmp_path)
    client = offline_client(store)
    first = execute_examples([example("train")], "", client, store, "original")
    second = execute_examples([example("train")], "", client, store, "original")
    assert first[0]["evaluation_id"] != second[0]["evaluation_id"]
    calls = client.ledger.summary()["calls"]
    with pytest.raises(ValueError, match="test execution"):
        execute_examples([example("test")], "", client, store, "original")
    assert client.ledger.summary()["calls"] == calls
    heldout = execute_examples([example("test")], "", client, store, "original", allow_test=True)
    assert heldout[0]["split"] == "test"


def test_budget_exhaustion_is_not_converted_into_a_wrong_answer(tmp_path):
    config = fixture_config()
    store = RunStore(tmp_path)
    client = offline_client(store, max_calls=1)
    with pytest.raises(BudgetExceeded):
        run_evolution(config, client, store)
    assert client.ledger.summary()["calls"] == 1


def test_invalid_optimizer_outputs_have_finite_attempts_and_scoped_patches_restore(tmp_path):
    ensure_upstream()
    import graphskillevo.model as model
    import graphskillevo.trainer as trainer

    class InvalidClient(OfflineClient):
        def generate(self, messages, **kwargs):
            generated = super().generate(messages, **kwargs)
            return type(generated)("{}", generated.prompt_tokens, generated.completion_tokens, 0)

    before = model.chat_optimizer, trainer.configure_models, trainer._get_token_summary
    config = fixture_config()
    config["evolution"]["max_operator_attempts"] = 2
    store = RunStore(tmp_path)
    client = InvalidClient(UsageLedger(max_calls=100, max_completion_tokens=100000))
    with pytest.raises(OperatorAttemptLimit, match="2 attempts"):
        run_evolution(config, client, store)
    assert client.ledger.summary()["calls"] == 2
    assert (model.chat_optimizer, trainer.configure_models, trainer._get_token_summary) == before


def test_failed_question_latency_covers_the_complete_client_call(tmp_path, monkeypatch):
    class FailingClient:
        def generate(self, *_args, **_kwargs):
            raise TimeoutError("both request attempts failed")

    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("repro.integration.qa_adapter.time.perf_counter", lambda: next(ticks))
    row = execute_examples([example()], "", FailingClient(), RunStore(tmp_path), "no_skill")[0]
    assert row["latency_s"] == 2.5
    assert row["status"] == "error"
    assert row["last_request_latency_s"] is None
    assert row["hard"] == 0


def test_evolution_dataloader_never_exposes_test_and_unbounded_model_calls_are_rejected(tmp_path):
    loader = QADataLoader([example("train", "training")], [example("val", "validation")], dataset="searchqa")
    with pytest.raises(ValueError, match="held out"):
        loader.build_eval_batch(0, "valid_unseen", 42)
    client = OfflineClient(UsageLedger())
    with pytest.raises(ValueError, match="finite model-call budget"):
        run_evolution(fixture_config(), client, RunStore(tmp_path))


def test_innovation_labels_cannot_silently_run_the_original_selector(tmp_path):
    config = fixture_config()
    config["experiment"]["condition"] = "improved"
    store = RunStore(tmp_path)
    client = offline_client(store)
    with pytest.raises(ValueError, match="registered selector"):
        run_evolution(config, client, store)
    assert client.ledger.summary()["calls"] == 0


def http_client(store, handler):
    settings = deepcopy(DEFAULTS["model"])
    settings["max_retries"] = 0
    return ChatClient(
        settings, UsageLedger(max_calls=10, sink=lambda row: store.append_jsonl("calls.jsonl", row)),
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.parametrize("status,body", [
    (401, {"error": {"message": "Authentication failed"}}),
    (403, {"error": {"message": "Forbidden model"}}),
    (404, {"error": {"code": "model_not_found"}}),
    (422, {"error": {"message": "Unsupported request field"}}),
    (400, {"error": {"code": "invalid_request_error"}}),
    (400, {"error": "Malformed request"}),
])
def test_fatal_http_errors_are_fully_recorded_and_stop_the_batch(tmp_path, status, body):
    store = RunStore(tmp_path)
    client = http_client(store, lambda _request: httpx.Response(status, json=body))
    try:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            execute_examples(
                [example(identifier="first"), example(identifier="second")],
                "", client, store, "no_skill",
            )
        assert raised.value.response.status_code == status
        assert client.ledger.summary()["calls"] == 1
        assert client.ledger.summary()["failed_calls"] == 1
        assert client.ledger.summary()["unknown_usage_calls"] == 1
        rows = [
            json.loads(line)
            for line in (store.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 1 and rows[0]["sample_id"] == "first" and rows[0]["status"] == "error"
        item_dir = store.path / rows[0]["evaluation_id"] / "predictions" / "first"
        request = json.loads((item_dir / "request.json").read_text(encoding="utf-8"))
        response = json.loads((item_dir / "response.json").read_text(encoding="utf-8"))
        assert request["messages"][1]["content"].endswith("Which city?")
        assert response["raw_response"] == {"status_code": status, "body": body}
        assert response["error"] and (item_dir / "result.json").is_file()
    finally:
        client.close()


def test_context_length_error_is_a_recorded_execution_failure_and_next_question_runs(tmp_path):
    store = RunStore(tmp_path)
    responses = iter([
        httpx.Response(400, json={"error": {"code": "context_length_exceeded"}}),
        httpx.Response(200, json={
            "choices": [{"message": {"content": "<answer>Paris</answer>"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }),
    ])
    client = http_client(store, lambda _request: next(responses))
    try:
        rows = execute_examples(
            [example(identifier="long"), example(identifier="short")], "", client, store, "no_skill",
        )
        assert [row["status"] for row in rows] == ["error", "ok"]
        assert [row["hard"] for row in rows] == [0, 1]
        assert client.ledger.summary()["calls"] == 2
        assert client.ledger.summary()["failed_calls"] == 1
    finally:
        client.close()


def test_evolution_rejects_a_batch_with_only_service_failures_after_recording_each(tmp_path):
    store = RunStore(tmp_path)
    client = http_client(
        store, lambda _request: httpx.Response(503, json={"error": "Temporarily unavailable"}),
    )
    adapter = QAAdapter(None, client, store, "original", 100)
    try:
        with pytest.raises(RuntimeError, match="All QA executions failed"):
            adapter.rollout(
                [example(identifier="first").to_dict(), example(identifier="second").to_dict()],
                "", str(store.path / "upstream" / "evaluation"),
            )
        rows = (store.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 2
        assert all(json.loads(row)["status"] == "error" for row in rows)
        assert client.ledger.summary()["calls"] == 2
    finally:
        client.close()


def test_evolution_keeps_a_completed_zero_em_batch_as_a_valid_skill_score(tmp_path):
    store = RunStore(tmp_path)
    client = offline_client(store)
    rows = QAAdapter(None, client, store, "original", 100).rollout(
        [example().to_dict()], "", str(store.path / "upstream" / "evaluation"),
    )
    assert rows[0]["status"] == "ok" and rows[0]["hard"] == 0


@pytest.mark.parametrize("condition", ["improved", "ablation"])
def test_registered_selector_switches_condition_and_is_recorded(tmp_path, monkeypatch, condition):
    from repro import selectors

    class FirstK:
        def __init__(self):
            self.calls = []

        def select(self, rows, *, k, seed):
            self.calls.append(k)
            return [row for row in rows if float(row.get("em", row.get("hard", 1))) < 1e-9][:k]

    placeholder = FirstK()
    monkeypatch.setattr(selectors, "_REGISTRY", dict(selectors._REGISTRY))
    selectors.register_selector(condition, placeholder)
    config = fixture_config()
    config["experiment"]["condition"] = condition
    config["evolution"]["selector"] = condition
    store = RunStore(tmp_path, condition)
    client = offline_client(store)
    run_evolution(config, client, store)
    assert (store.path / "upstream/best_skill.md").is_file()
    events = [json.loads(line) for line in (store.path / "events.jsonl").read_text(
        encoding="utf-8").splitlines()]
    batches = [event for event in events if event["event"] == "training_failure_batch"]
    # The bridge records only batches that feed the optimizer (k > 0); k == 0 calls are not logged.
    used = [k for k in placeholder.calls if k > 0]
    assert used and len(batches) == len(used)
    assert {event["selector"] for event in batches} == {condition}
    assert all(set(event["selected_ids"]) <= set(event["source_ids"]) for event in batches)
    assert all(len(event["selected_ids"]) <= config["evolution"]["reflection_failure_k"] for event in batches)


def test_unregistered_selector_name_fails_before_any_model_call(tmp_path):
    config = fixture_config()
    config["experiment"]["condition"] = "improved"
    config["evolution"]["selector"] = "improved"
    store = RunStore(tmp_path)
    client = offline_client(store)
    with pytest.raises(NotImplementedError, match="member B"):
        run_evolution(config, client, store)
    assert client.ledger.summary()["calls"] == 0
