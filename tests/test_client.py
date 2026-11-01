"""Request boundary failures must remain visible in accounting."""

from copy import deepcopy

import httpx
import pytest

from repro.config import DEFAULTS
from repro.inference.accounting import BudgetExceeded, UsageLedger
from repro.inference.client import ChatClient


def test_usage_missing_is_not_counted_as_zero():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "<answer>Paris</answer>"}, "finish_reason": "stop"}],
            },
        )
    )
    ledger = UsageLedger(max_calls=2, max_completion_tokens=100)
    client = ChatClient(deepcopy(DEFAULTS["model"]), ledger, transport=transport)
    result = client.generate([{"role": "user", "content": "test"}], max_tokens=20)
    assert result.prompt_tokens is None
    assert not ledger.summary()["usage_complete"]
    assert ledger.summary()["charged_completion_tokens"] == 20
    client.close()


def test_server_error_retry_counts_both_attempts():
    replies = iter(
        [
            httpx.Response(503),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "yes"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            ),
        ]
    )
    ledger = UsageLedger(max_calls=2, max_completion_tokens=100)
    client = ChatClient(
        deepcopy(DEFAULTS["model"]), ledger, transport=httpx.MockTransport(lambda request: next(replies))
    )
    assert client.generate([{"role": "user", "content": "test"}], max_tokens=20).content == "yes"
    assert ledger.summary()["calls"] == 2
    assert ledger.summary()["failed_calls"] == 1
    client.close()


def test_budget_blocks_network_request():
    ledger = UsageLedger(max_calls=0)
    client = ChatClient(
        deepcopy(DEFAULTS["model"]),
        ledger,
        transport=httpx.MockTransport(lambda request: pytest.fail("Network called")),
    )
    with pytest.raises(BudgetExceeded):
        client.generate([{"role": "user", "content": "test"}])
    client.close()


def test_service_model_and_thinking_mode_must_match():
    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": DEFAULTS["model"]["name"]}]})
        return httpx.Response(200, json={"backend": "transformers", "enable_thinking": True})

    client = ChatClient(deepcopy(DEFAULTS["model"]), UsageLedger(), transport=httpx.MockTransport(respond))
    with pytest.raises(ValueError, match="thinking mode"):
        client.verify_service()
    assert client.ledger.summary()["calls"] == 0
    client.close()


def test_llama_cpp_uses_supported_max_tokens_field():
    import json

    settings = deepcopy(DEFAULTS["model"])
    settings["backend"] = "llama_cpp"

    def respond(request):
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 17
        assert "max_completion_tokens" not in payload
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    client = ChatClient(settings, UsageLedger(), transport=httpx.MockTransport(respond))
    assert client.generate([{"role": "user", "content": "test"}], max_tokens=17).content == "answer"
    client.close()


def test_llama_service_properties_are_saved_and_wrong_context_rejected():
    from repro.config import PROJECT_ROOT
    from repro.serving.llama_runtime import read_runtime_lock

    settings = deepcopy(DEFAULTS["model"])
    settings["backend"] = "llama_cpp"
    lock = read_runtime_lock(PROJECT_ROOT)
    context = 8192

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": settings["name"]}]})
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "model_alias": settings["name"],
                    "total_slots": 1,
                    "build_info": f"{lock['release']}-{lock['commit'][:9]}",
                    "default_generation_settings": {"n_ctx": context},
                },
            )
        return httpx.Response(200, json={"status": "ok"})

    client = ChatClient(settings, UsageLedger(), transport=httpx.MockTransport(respond))
    report = client.verify_service()
    assert report["properties"]["reported_context_tokens"] == 8192
    context = 4096
    with pytest.raises(ValueError, match="context"):
        client.verify_service()
    assert client.ledger.summary()["calls"] == 0
    client.close()


@pytest.mark.parametrize(
    "health_override,match",
    [({"max_model_len": 4096}, "context"), ({"local_path": "different-model"}, "model path")],
)
def test_transformers_service_cannot_silently_use_other_settings(health_override, match):
    settings = deepcopy(DEFAULTS["model"])

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": settings["name"]}]})
        return httpx.Response(200, json={
            "backend": "transformers", "enable_thinking": settings["enable_thinking"],
            "max_model_len": settings["max_context_tokens"], "local_path": settings["local_path"],
            **health_override,
        })

    client = ChatClient(settings, UsageLedger(), transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(ValueError, match=match):
            client.verify_service()
        assert client.ledger.summary()["calls"] == 0
    finally:
        client.close()
