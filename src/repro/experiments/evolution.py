"""Run the adapted official optimizer inside a traceable experiment context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repro.artifacts.gpu import GPUMonitor
from repro.experiments.context import finish_run, prepare_run, verify_run_service


def run_evolution_experiment(config: dict[str, Any], *, offline: bool = False) -> Path:
    from repro.integration.graphskillevo import run_evolution

    if config["experiment"]["condition"] not in {"original", "improved", "ablation"}:
        raise ValueError("Evolution requires original, improved, or ablation condition")
    store, client = prepare_run(config, offline=offline)
    labels = {"synthetic": config["experiment"]["profile"] == "synthetic", "backend_offline": offline}
    try:
        verify_run_service(store, client, offline=offline)
        with GPUMonitor(lambda row: store.append_jsonl("gpu.jsonl", row)):
            result = run_evolution(config, client, store)
        result.update(labels)
    except Exception as exc:
        finish_run(store, client, labels, error=exc)
        raise
    finish_run(store, client, result)
    return store.path
