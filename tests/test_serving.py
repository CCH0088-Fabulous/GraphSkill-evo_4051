from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from repro.serving.app import create_app
from repro.serving.model import Completion, ContextLimitError, LocalTextModel


class FakeBackend:
    model_name = "fixture-model"
    max_model_len = 128
    enable_thinking = False

    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs["max_new_tokens"] > self.max_model_len:
            raise ContextLimitError("Request exceeds fixture context")
        return Completion("fixture answer", prompt_tokens=17, completion_tokens=3, finish_reason="stop")


def test_openai_response_usage_and_token_limit_forwarding():
    backend = FakeBackend()
    client = TestClient(create_app(backend))
    assert client.get("/v1/models").json()["data"][0]["id"] == "fixture-model"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "Question"}],
            "max_completion_tokens": 11,
        },
    )
    assert response.status_code == 200
    assert response.json()["usage"] == {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20}
    assert response.json()["choices"][0]["message"]["content"] == "fixture answer"
    assert backend.calls[0][1]["max_new_tokens"] == 11


def test_actual_openai_sdk_parses_local_response():
    from openai import OpenAI

    http_client = TestClient(create_app(FakeBackend()))
    with OpenAI(api_key="fixture", base_url="http://testserver/v1", http_client=http_client) as client:
        result = client.chat.completions.create(
            model="fixture-model", messages=[{"role": "user", "content": "Question"}], max_completion_tokens=7
        )
    assert result.choices[0].message.content == "fixture answer"
    assert result.usage.prompt_tokens == 17
    assert result.usage.completion_tokens == 3


def test_metered_project_client_matches_native_server_protocol():
    from repro.inference.accounting import UsageLedger
    from repro.inference.client import ChatClient

    backend = FakeBackend()
    app_client = TestClient(create_app(backend))

    def route(request):
        response = app_client.request(
            request.method,
            request.url.path,
            content=request.content,
            headers={"Content-Type": "application/json"},
        )
        return httpx.Response(response.status_code, json=response.json())

    ledger = UsageLedger(max_calls=5, max_completion_tokens=100)
    settings = {
        "name": "fixture-model",
        "base_url": "http://testserver/v1",
        "api_key_env": "REPRO_API_KEY",
        "timeout_s": 10,
        "max_completion_tokens": 11,
        "temperature": 0.0,
        "enable_thinking": False,
        "max_retries": 0,
    }
    chat = ChatClient(settings, ledger, transport=httpx.MockTransport(route))
    try:
        generation = chat.generate([{"role": "user", "content": "Question"}])
    finally:
        chat.close()
        app_client.close()
    assert generation.content == "fixture answer"
    assert generation.completion_tokens == 3
    assert ledger.summary()["calls"] == 1
    assert ledger.summary()["charged_completion_tokens"] == 3


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"n": 2},
        {"reasoning_effort": "medium"},
        {"max_tokens": 3, "max_completion_tokens": 4},
        {"chat_template_kwargs": {"enable_thinking": True}},
    ],
)
def test_unsupported_modes_fail_without_generating(extra):
    backend = FakeBackend()
    response = TestClient(create_app(backend)).post(
        "/v1/chat/completions",
        json={
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "Question"}],
            **extra,
        },
    )
    assert response.status_code == 400
    assert not backend.calls


def test_unknown_chat_template_options_are_rejected():
    backend = FakeBackend()
    response = TestClient(create_app(backend)).post(
        "/v1/chat/completions",
        json={
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "Question"}],
            "chat_template_kwargs": {"enable_thinking": False, "unknown_option": 1},
        },
    )
    assert response.status_code == 422
    assert not backend.calls


def test_context_error_is_a_structured_client_error():
    response = TestClient(create_app(FakeBackend())).post(
        "/v1/chat/completions",
        json={
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "Question"}],
            "max_completion_tokens": 1000,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_missing_local_model_is_rejected_before_loading_libraries(tmp_path):
    with pytest.raises(ValueError, match="Incomplete local model"):
        LocalTextModel.from_directory(tmp_path / "not-a-model")


def test_model_counts_generated_ids_and_rejects_context_without_truncation():
    torch = pytest.importorskip("torch")

    class Inputs(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def apply_chat_template(self, _messages, **kwargs):
            self.options = kwargs
            return Inputs(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))

        def decode(self, _tokens, **_kwargs):
            return "answer"

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=99)

        def generate(self, **kwargs):
            self.options = kwargs
            return torch.tensor([[1, 2, 3, 4, 5, 11, 12, 99]])

    tokenizer, model = Tokenizer(), Model()
    backend = LocalTextModel(
        model=model, tokenizer=tokenizer, torch_module=torch, model_name="fixture", max_model_len=10
    )
    result = backend.generate([{"role": "user", "content": "q"}], max_new_tokens=5)
    assert (result.prompt_tokens, result.completion_tokens, result.finish_reason) == (5, 3, "stop")
    assert model.options["pad_token_id"] == 0
    assert tokenizer.options["enable_thinking"] is False
    assert "truncation" not in tokenizer.options
    with pytest.raises(ContextLimitError, match="No input was truncated"):
        backend.generate([{"role": "user", "content": "q"}], max_new_tokens=6)
