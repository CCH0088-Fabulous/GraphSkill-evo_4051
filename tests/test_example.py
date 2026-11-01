"""The one-example check must execute only the first fixed training example."""

import json
from contextlib import contextmanager

import httpx

from repro.config import load_config
from repro.datasets.io import load_split
from repro.experiments.example import run_example
from repro.inference.client import ChatClient


def test_example_uses_training_only_and_is_labeled_as_engineering(tmp_path, monkeypatch):
    cfg = load_config("configs/offline.searchqa.yaml", [
        "model.backend=transformers", "experiment.condition=initial_skill",
        f"experiment.output_root={tmp_path.as_posix()}",
    ])
    calls, splits = [], []

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": cfg["model"]["name"]}]})
        if request.url.path == "/health":
            return httpx.Response(200, json={"backend": "transformers", "enable_thinking": False})
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "<answer>Saturn</answer>"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7},
        })

    def tracked_split(path, split, dataset):
        splits.append(split)
        return load_split(path, split, dataset)

    @contextmanager
    def fixture_server(_settings, _path):
        yield ["fixture-server"]

    @contextmanager
    def fixture_monitor(sink, interval_s):
        sink({"event": "gpu_sample", "memory_used_mib": 1})
        yield

    monkeypatch.setattr("repro.experiments.example.load_split", tracked_split)
    monkeypatch.setattr("repro.experiments.example.managed_transformers_server", fixture_server)
    monkeypatch.setattr("repro.experiments.example.GPUMonitor", fixture_monitor)
    monkeypatch.setattr("repro.experiments.context.ChatClient", lambda settings, ledger, seed: ChatClient(
        settings, ledger, seed=seed, transport=httpx.MockTransport(respond),
    ))
    run = run_example(cfg)
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    prediction = json.loads((run / "predictions.jsonl").read_text(encoding="utf-8"))
    assert splits == ["train"]
    assert len(calls) == summary["usage"]["calls"] == 1
    assert summary["purpose"] == "engineering_example" and summary["research_results"] is False
    assert summary["split"] == "train" and summary["n"] == 1
    assert prediction["sample_id"] == "synthetic-searchqa-train-004"
