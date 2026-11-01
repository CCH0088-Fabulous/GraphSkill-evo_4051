"""Run normalized QA records and expose an upstream-compatible environment adapter."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from repro.artifacts import RunStore
from repro.contracts import QAExample
from repro.evaluation.metrics import evaluate
from repro.inference.accounting import BudgetExceeded
from repro.integration.source import ensure_upstream


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value.endswith("."):
        raise ValueError(f"Example ID cannot be used safely as an artifact directory: {value!r}")
    if value.split(".", 1)[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError(f"Reserved filesystem example ID: {value!r}")
    return value


def build_prompts(example: QAExample, skill_content: str) -> list[dict[str, str]]:
    """Keep the original SearchQA prompt/truncation; retain all HotpotQA passages."""
    ensure_upstream()
    from graphskillevo.envs.searchqa.rollout import _build_system, _build_user

    system = _build_system(skill_content)
    user = (
        _build_user(example.question, example.context)
        if example.dataset == "searchqa"
        else f"## Context\n{example.context}\n\n## Question\n{example.question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def execute_examples(
    examples: list[QAExample],
    skill_content: str,
    client: Any,
    store: RunStore,
    condition: str,
    *,
    skill_id: str | None = None,
    out_dir: str | Path | None = None,
    allow_test: bool = False,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Execute one evaluation with raw evidence and upstream reflection artifacts.

    Execution errors are kept as failed predictions. Authentication, model, and
    protocol HTTP errors are recorded before propagating to stop the batch.
    Budget exhaustion propagates immediately rather than becoming a QA answer.
    Each invocation gets an evaluation ID so repeat training evaluations of a
    retained individual remain separate observations.
    """
    if not examples:
        return []
    if any(example.split == "test" for example in examples) and not allow_test:
        raise ValueError("Held-out test execution requires explicit experiment.allow_test=true")
    if len({(example.dataset, example.split) for example in examples}) != 1:
        raise ValueError("One evaluation must contain a single dataset and split")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("An evaluation contains duplicate example IDs")
    for example in examples:
        _safe_id(example.id)
    skill_id = skill_id or hashlib.sha256(skill_content.encode()).hexdigest()
    target = Path(out_dir) if out_dir is not None else store.path / "evaluations" / uuid.uuid4().hex
    if not target.is_absolute():
        target = store.path / target
    target = target.resolve()
    if not target.is_relative_to(store.path.resolve()):
        raise ValueError("Evaluation output directory must be inside its run directory")
    relative = target.relative_to(store.path.resolve())
    evaluation_id = relative.as_posix()
    results = []
    for example in examples:
        messages = build_prompts(example, skill_content)
        item_dir = relative / "predictions" / example.id
        store.save_text(item_dir / "target_system_prompt.txt", messages[0]["content"])
        store.save_text(item_dir / "target_user_prompt.txt", messages[1]["content"])
        store.save_json(item_dir / "example.json", example.to_dict())
        output, error, status, response = "", None, "ok", {}
        fatal_error = None
        last_request_latency = None
        finish_reason = None
        call_started = time.perf_counter()
        try:
            generation = client.generate(
                messages,
                role="target",
                max_tokens=max_tokens,
                metadata={
                    "sample_id": example.id,
                    "dataset": example.dataset,
                    "split": example.split,
                    "condition": condition,
                    "skill_id": skill_id,
                    "evaluation_id": evaluation_id,
                },
            )
            output, response = generation.content, generation.raw_response
            last_request_latency = generation.latency_s
            finish_reason = generation.finish_reason
            if finish_reason == "length":
                status = "truncated"
        except BudgetExceeded:
            store.record_event(
                "evaluation_budget_exhausted",
                evaluation_id=evaluation_id,
                sample_id=example.id,
            )
            raise
        except Exception as exc:
            error, status = f"{type(exc).__name__}: {exc}", "error"
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    body = exc.response.json()
                except (ValueError, UnicodeError):
                    body = exc.response.text
                response = {"status_code": exc.response.status_code, "body": body}
                details = body.get("error") if isinstance(body, dict) else None
                code = details.get("code") if isinstance(details, dict) else None
                fatal = exc.response.status_code in {401, 403, 404, 422} or (
                    exc.response.status_code == 400 and code != "context_length_exceeded"
                )
                if fatal:
                    fatal_error = exc
            store.record_event("target_error", evaluation_id=evaluation_id, sample_id=example.id, error=error)
        latency = time.perf_counter() - call_started
        scores = evaluate("" if status == "truncated" else output, example.answers, dataset=example.dataset)
        if status != "ok":
            scores.update(em=0.0, f1=0.0)
        row = {
            "id": example.id,
            "sample_id": example.id,
            "dataset": example.dataset,
            "split": example.split,
            "condition": condition,
            "skill_id": skill_id,
            "evaluation_id": evaluation_id,
            "question": example.question,
            "task_description": example.question,
            "task_type": "qa",
            "gold_answers": list(example.answers),
            "answer": scores["answer"],
            "predicted_answer": scores["answer"],
            "em": scores["em"],
            "f1": scores["f1"],
            "hard": int(scores["em"]),
            "soft": scores["f1"],
            "response": output,
            "raw_output": output,
            "latency_s": latency,
            "last_request_latency_s": last_request_latency,
            "finish_reason": finish_reason,
            "output_truncated": status == "truncated",
            "failure_type": "output_truncated"
            if status == "truncated"
            else ("execution_error" if error else None),
            "status": status,
            "error": error,
            "agent_ok": status == "ok",
            "n_turns": int(status != "error"),
            "target_system_prompt": messages[0]["content"],
            "target_user_prompt": messages[1]["content"],
            "fail_reason": error
            or (
                "Output truncated at the fixed token limit; incomplete answer scored zero."
                if status == "truncated"
                else (
                    f"EM=0: predicted '{scores['answer']}' but expected {list(example.answers)}"
                    if scores["em"] == 0
                    else ""
                )
            ),
        }
        # Preserve the original SearchQA reflection transcript shape. Prompt files
        # are read separately by the upstream formatter, so do not duplicate them
        # inside the trajectory or add a new Hidden Reference optimization input.
        evaluation_detail = (
            f"[EVALUATION RESULT]\nQuestion: {example.question}\n"
            f"Predicted answer: {scores['answer']!r}\nGold answers: {list(example.answers)!r}\n"
            f"Exact Match: {scores['em']}\nF1: {scores['f1']:.4f}"
        )
        if status == "truncated":
            evaluation_detail += "\nFailure type: output_truncated at the fixed generation limit."
        conversation = [
            {"type": "message", "turn": 1, "content": output},
            {"role": "system", "content": evaluation_detail},
        ]
        store.save_json(item_dir / "request.json", {"messages": messages})
        store.save_json(item_dir / "conversation.json", conversation)
        store.save_json(item_dir / "response.json", {"raw_response": response, "error": error})
        store.save_json(item_dir / "result.json", row)
        store.write_prediction(row)
        results.append(row)
        if fatal_error is not None:
            store.record_event(
                "model_configuration_error",
                evaluation_id=evaluation_id,
                sample_id=example.id,
                status_code=fatal_error.response.status_code,
                error=error,
            )
            raise fatal_error
    return results


class QAAdapter:
    """Duck-typed upstream EnvAdapter that owns QA execution, not data selection."""

    def __init__(
        self,
        dataloader: Any,
        client: Any,
        store: RunStore,
        condition: str,
        max_tokens: int,
    ) -> None:
        self.dataloader, self.client, self.store = dataloader, client, store
        self.condition, self.max_tokens = condition, max_tokens

    def setup(self, cfg: dict[str, Any]) -> None:
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch, **kwargs):
        return list(batch.payload or [])

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict[str, Any]]:
        from repro.datasets.io import example_from_record

        examples = [example_from_record(item) for item in env_manager]
        # Evolution never receives test examples, even if final evaluation is enabled in the run config.
        rows = execute_examples(
            examples,
            skill_content,
            self.client,
            self.store,
            self.condition,
            out_dir=out_dir,
            max_tokens=self.max_tokens,
            allow_test=False,
        )
        if not rows or all(row["status"] == "error" for row in rows):
            raise RuntimeError(
                "All QA executions failed in this batch; unavailable model service cannot be a skill score"
            )
        return rows

    def build_reference_text(self, item: dict[str, Any]) -> str:
        if item["split"] != "train":
            return ""
        return str(item.get("reference_text") or "").strip()

    def get_task_types(self) -> list[str]:
        return ["qa"]
