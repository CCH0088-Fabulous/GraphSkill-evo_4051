"""Temporarily route upstream optimizers and counters through one local client."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from repro.artifacts import RunStore
from repro.inference.seeding import optimizer_retry_seed, optimizer_seed
from repro.selectors import get_selector

_UPSTREAM_LOCK = threading.RLock()


class OperatorAttemptLimit(RuntimeError):
    """Invalid graph generations exhausted the explicitly bounded operation phase."""


class OptimizerBridge:
    def __init__(self, client: Any, store: RunStore, *, seed_policy: str = "fixed") -> None:
        self.client, self.store = client, store
        self.seed_policy = seed_policy
        self.calls = 0
        self._scope: str | None = None
        self._limit = 0
        self._attempts = 0
        self._stage_attempts: dict[str, int] = {}

    @contextlib.contextmanager
    def phase(self, name: str, max_attempts: int) -> Iterator[None]:
        previous = (self._scope, self._limit, self._attempts, self._stage_attempts)
        self._scope, self._limit, self._attempts = name, max_attempts, 0
        self._stage_attempts = {}
        try:
            yield
        finally:
            self._scope, self._limit, self._attempts, self._stage_attempts = previous

    def chat_optimizer(
        self,
        system: str,
        user: str,
        max_completion_tokens: int = 4096,
        retries: int = 3,
        stage: str = "optimizer",
        reasoning_effort=None,
        timeout=None,
    ) -> tuple[str, dict[str, Any]]:
        if self._scope is None:
            raise RuntimeError("Upstream optimizer invoked outside a bounded integration phase")
        while True:
            if self._attempts >= self._limit:
                raise OperatorAttemptLimit(f"Operator phase '{self._scope}' exhausted {self._limit} attempts")
            self._attempts += 1
            stage_attempt = self._stage_attempts.get(stage, 0) + 1
            self._stage_attempts[stage] = stage_attempt
            self.calls += 1
            base_seed = getattr(self.client, "seed", 42)
            request_seed = (
                optimizer_seed(base_seed, self._scope, stage, self._attempts)
                if self.seed_policy == "optimizer_attempt_v1"
                else base_seed
            )
            if self.seed_policy == "optimizer_retry_v1":
                request_seed = optimizer_retry_seed(base_seed, self._scope, stage, stage_attempt)
            record: dict[str, Any] = {
                "stage": stage,
                "phase": self._scope,
                "attempt": self._attempts,
                "system": system,
                "user": user,
                "seed": request_seed,
                "seed_policy": self.seed_policy,
                "seed_attempt": stage_attempt,
                "upstream_retries_argument": retries,
                "retry_owner": "repro.inference.client",
            }
            try:
                result = self.client.generate(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    role="optimizer",
                    max_tokens=max_completion_tokens,
                    seed=request_seed,
                    metadata={
                        "stage": stage,
                        "phase": self._scope,
                        "operator_attempt": self._attempts,
                        "seed_attempt": stage_attempt,
                        "seed_policy": self.seed_policy,
                    },
                )
                usage = {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": (
                        result.prompt_tokens + result.completion_tokens
                        if result.prompt_tokens is not None and result.completion_tokens is not None
                        else None
                    ),
                }
                truncated = result.finish_reason == "length"
                record.update(
                    raw_response=result.content,
                    usage=usage,
                    finish_reason=result.finish_reason,
                    output_truncated=truncated,
                )
                if truncated:
                    self.store.record_event(
                        "optimizer_output_truncated",
                        stage=stage,
                        phase=self._scope,
                        operator_attempt=self._attempts,
                        seed=request_seed,
                    )
                    continue
                return result.content, usage
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.store.save_json(f"operators/call_{self.calls:05d}.json", record)

    def token_summary(self) -> dict[str, Any]:
        summary = self.client.ledger.summary()
        by_role = {
            role: {
                "calls": counts["calls"],
                "prompt_tokens": counts["prompt_tokens"],
                "completion_tokens": counts["completion_tokens"],
                "total_tokens": counts["prompt_tokens"] + counts["completion_tokens"],
            }
            for role, counts in summary.get("by_role", {}).items()
        }
        return {
            **by_role,
            "_total": {
                "calls": summary["calls"],
                "prompt_tokens": summary["prompt_tokens"],
                "completion_tokens": summary["completion_tokens"],
                "total_tokens": summary["prompt_tokens"] + summary["completion_tokens"],
                "usage_complete": summary["usage_complete"],
                "unknown_usage_calls": summary["unknown_usage_calls"],
            },
        }


@contextlib.contextmanager
def scoped_upstream(bridge: OptimizerBridge, *, selector_name: str, dataset: str) -> Iterator[None]:
    """Restore every upstream global after success or failure; serialize patches."""
    import graphskillevo.evaluation as evaluation
    import graphskillevo.model as model
    import graphskillevo.trainer as trainer

    selector = get_selector(selector_name)
    original_task_context = trainer.build_initial_task_context

    def sample_failures(rows, *, failure_k, seed):
        if failure_k > 0 and any(row.get("split") != "train" for row in rows):
            raise ValueError("Only training failures may be selected for optimizer feedback")
        selected = selector.select(rows, k=failure_k, seed=seed)
        if failure_k > 0:
            bridge.store.record_event(
                "training_failure_batch",
                selector=selector_name,
                seed=seed,
                source_ids=[row["id"] for row in rows],
                selected_ids=[row["id"] for row in selected],
            )
        return selected

    def task_context(cfg):
        if dataset == "hotpotqa":
            return (
                "HotpotQA: answer a question from supplied titled passages, connecting evidence across "
                "passages or comparing entities when needed. Return a short answer in <answer> tags. "
                "Reference answers and supporting-fact annotations are not task inputs."
            )
        return original_task_context(cfg)

    with _UPSTREAM_LOCK, contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(model, "chat_optimizer", bridge.chat_optimizer))
        stack.enter_context(patch.object(trainer, "configure_models", lambda cfg: None))
        stack.enter_context(patch.object(trainer, "_get_token_summary", bridge.token_summary))
        stack.enter_context(patch.object(trainer, "_reset_token_tracker", lambda: None))
        stack.enter_context(patch.object(trainer, "build_initial_task_context", task_context))
        stack.enter_context(patch.object(evaluation, "_sample_failure_rows", sample_failures))
        yield
