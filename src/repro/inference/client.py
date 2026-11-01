"""One metered OpenAI-compatible HTTP client shared by every model role."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from repro.contracts import Generation
from repro.inference.accounting import UsageLedger
from repro.inference.service import validate_llama_properties


class ChatClient:
    def __init__(
        self,
        settings: dict[str, Any],
        ledger: UsageLedger,
        *,
        seed: int = 42,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.seed = seed
        self._http = httpx.Client(
            base_url=settings["base_url"].rstrip("/") + "/",
            timeout=settings["timeout_s"],
            transport=transport,
            headers={"Authorization": f"Bearer {os.environ.get(settings['api_key_env'], 'local-only')}"},
            trust_env=False,
        )

    def close(self) -> None:
        self._http.close()

    def models(self) -> dict[str, Any]:
        response = self._http.get("models")
        response.raise_for_status()
        return response.json()

    def verify_service(self) -> dict[str, Any]:
        """Fail before an experiment if its model or fixed thinking mode differs."""
        try:
            models = self.models()
            names = [item["id"] for item in models.get("data", [])]
            if self.settings["name"] not in names:
                raise ValueError(f"Configured model is not served: {self.settings['name']}")
            health_url = httpx.URL(self.settings["base_url"]).copy_with(path="/health")
            response = self._http.get(health_url)
            health = response.json() if response.status_code == 200 and response.content else {}
            if health.get("backend") == "transformers":
                if health.get("enable_thinking") != self.settings["enable_thinking"]:
                    raise ValueError("Configured thinking mode differs from the running local server")
                if health.get("max_model_len") is not None:
                    if health["max_model_len"] != self.settings["max_context_tokens"]:
                        raise ValueError("Configured context differs from the running local server")
                if health.get("local_path") is not None:
                    from repro.config import resolve_path

                    if resolve_path(health["local_path"]) != resolve_path(self.settings["local_path"]):
                        raise ValueError("Configured model path differs from the running local server")
            result = {"models": names, "health": health}
            if self.settings.get("backend") == "llama_cpp":
                props_url = health_url.copy_with(path="/props")
                props_response = self._http.get(props_url)
                props_response.raise_for_status()
                from repro.config import PROJECT_ROOT
                from repro.serving.llama_runtime import read_runtime_lock

                lock = read_runtime_lock(PROJECT_ROOT)
                expected_build = f"{lock['release']}-{lock['commit'][:9]}"
                result["properties"] = validate_llama_properties(
                    props_response.json(),
                    self.settings,
                    expected_build=expected_build,
                )
            return result
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Model service is unavailable. Start scripts/serve_gguf.ps1 for llama_cpp "
                "or scripts/serve_model.ps1 for Transformers first."
            ) from exc

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        role: str = "target",
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Generation:
        limit = max_tokens or self.settings["max_completion_tokens"]
        payload = {
            "model": self.settings["name"],
            "messages": messages,
            "temperature": self.settings["temperature"],
            "seed": self.seed if seed is None else seed,
            "top_p": self.settings.get("top_p", 1.0),
            "top_k": self.settings.get("top_k", 50),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.settings["enable_thinking"]},
        }
        limit_field = "max_tokens" if self.settings.get("backend") == "llama_cpp" else "max_completion_tokens"
        payload[limit_field] = limit
        for attempt in range(self.settings["max_retries"] + 1):
            call_id = self.ledger.begin_call(role, limit, {**(metadata or {}), "http_attempt": attempt + 1})
            started = time.perf_counter()
            try:
                response = self._http.post("chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                content = choice["message"].get("content")
                if not isinstance(content, str):
                    raise ValueError("Backend did not return textual assistant content")
                usage = body.get("usage") or {}
                prompt_tokens = _token_count(usage.get("prompt_tokens"))
                completion_tokens = _token_count(usage.get("completion_tokens"))
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                self.ledger.finish_call(
                    call_id,
                    prompt_tokens=None,
                    completion_tokens=None,
                    latency_s=time.perf_counter() - started,
                    status="error",
                    error=str(exc),
                    request=payload,
                    response_body=(exc.response.text if isinstance(exc, httpx.HTTPStatusError) else None),
                )
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and (exc.response.status_code == 429 or exc.response.status_code >= 500)
                )
                if not retryable or attempt == self.settings["max_retries"]:
                    raise
                time.sleep(min(0.5 * 2**attempt, 4))
                continue
            elapsed = time.perf_counter() - started
            self.ledger.finish_call(
                call_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=elapsed,
                request=payload,
                response=body,
            )
            return Generation(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_s=elapsed,
                finish_reason=choice.get("finish_reason"),
                response_id=body.get("id"),
                raw_response=body,
            )
        raise RuntimeError("Unreachable retry state")


def _token_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
