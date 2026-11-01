"""A small non-streaming OpenAI-compatible HTTP boundary for native Windows inference."""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from repro.serving.model import Completion, ContextLimitError, LocalTextModel

logger = logging.getLogger(__name__)


class TextBackend(Protocol):
    model_name: str
    max_model_len: int
    enable_thinking: bool

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None,
    ) -> Completion: ...


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatTemplateOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable_thinking: StrictBool


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=50, ge=0)
    stream: bool = False
    n: int = Field(default=1, ge=1)
    seed: int | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: ChatTemplateOptions | None = None


def _error(message: str, *, status: int = 400, code: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": code, "code": code}}
    )


def create_app(backend: TextBackend) -> FastAPI:
    """The backend is injected so protocol tests never require model weights or a GPU."""
    app = FastAPI(title="GraphSkill local model server", docs_url=None, redoc_url=None)
    started = int(time.time())

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return _error("Unsupported or invalid request fields", status=422)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "model": backend.model_name,
            "backend": "transformers",
            "max_model_len": backend.max_model_len,
            "enable_thinking": backend.enable_thinking,
            "local_path": str(backend.local_path) if getattr(backend, "local_path", None) else None,
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": backend.model_name,
                    "object": "model",
                    "created": started,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def complete(body: ChatRequest) -> Any:
        if body.model != backend.model_name:
            return _error("Requested model is not served", status=404, code="model_not_found")
        if body.stream or body.n != 1:
            return _error("Only stream=false and n=1 are supported")
        if body.reasoning_effort not in (None, "none"):
            return _error("reasoning_effort cannot vary; thinking mode is fixed when the server starts")
        if body.chat_template_kwargs is not None:
            if body.chat_template_kwargs.enable_thinking != backend.enable_thinking:
                return _error("Requested enable_thinking differs from the fixed server thinking mode")
        if body.max_tokens is not None and body.max_completion_tokens is not None:
            return _error("Specify only one of max_tokens and max_completion_tokens")
        limit = body.max_completion_tokens or body.max_tokens or 1024
        try:
            result = backend.generate(
                [message.model_dump() for message in body.messages],
                max_new_tokens=limit,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                seed=body.seed,
            )
        except ContextLimitError as exc:
            return _error(str(exc), code="context_length_exceeded")
        except ValueError as exc:
            return _error(str(exc))
        except RuntimeError:
            logger.exception("Local model generation failed")
            return _error(
                "Local generation failed; inspect the server terminal", status=500, code="server_error"
            )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": backend.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    backend = LocalTextModel.from_directory(
        args.model_path,
        model_name=args.model_name,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )
    import uvicorn

    uvicorn.run(create_app(backend), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
