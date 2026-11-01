"""Compose one metered training-example check, separate from research experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from repro.artifacts.gpu import GPUMonitor
from repro.config import resolve_path
from repro.datasets.io import load_split
from repro.experiments.context import finish_run, prepare_run, skill_path, verify_run_service
from repro.integration.qa_adapter import execute_examples
from repro.serving.lifecycle import managed_transformers_server


def run_example(config: dict[str, Any]) -> Path:
    if config["model"]["backend"] != "transformers":
        raise ValueError("This example requires model.backend=transformers")
    if config["experiment"]["condition"] != "initial_skill":
        raise ValueError("The example uses the initial_skill condition")
    examples = load_split(resolve_path(config["data"]["split_dir"]), "train", config["data"]["dataset"])
    if not examples:
        raise ValueError("No training example available")
    print("Freezing local model hashes and run provenance (no downloads).", flush=True)
    store, client = prepare_run(config)
    labels = {
        "purpose": "engineering_example", "research_results": False,
        "synthetic": config["experiment"]["profile"] == "synthetic",
        "backend_offline": False, "dataset": config["data"]["dataset"], "split": "train", "n": 1,
    }
    started = time.perf_counter()
    try:
        content = skill_path(config).read_text(encoding="utf-8")
        skill = store.save_skill("initial", content, condition="initial_skill")
        with GPUMonitor(lambda row: store.append_jsonl("gpu.jsonl", row), interval_s=1):
            with managed_transformers_server(config["model"], store.path / "server") as argv:
                store.save_json("server_command.json", {"argv": argv})
                verify_run_service(store, client, offline=False)
                ready_s = time.perf_counter() - started
                print("Model ready; executing the first frozen training example.", flush=True)
                rows = execute_examples(
                    examples[:1], content, client, store, "initial_skill", skill_id=skill["sha256"],
                )
                if rows[0]["status"] != "ok":
                    raise RuntimeError(f"Example generation failed: {rows[0]['error']}")
        samples = [json.loads(line) for line in (store.path / "gpu.jsonl").read_text().splitlines()]
        memory = [row["memory_used_mib"] for row in samples if row.get("memory_used_mib") is not None]
        result = {
            **labels, "question": examples[0].question, "gold_answers": list(examples[0].answers),
            "answer": rows[0]["answer"], "em": rows[0]["em"], "f1": rows[0]["f1"],
            "request_latency_s": rows[0]["last_request_latency_s"],
            "server_startup_s": ready_s, "total_wall_time_s": time.perf_counter() - started,
            "observed_peak_gpu_total_memory_mib": max(memory) if memory else None,
            "gpu_memory_note": "Device-wide sampled usage, including other applications; 1 second interval.",
            "temporary_server_stopped": True,
        }
    except Exception as exc:
        finish_run(store, client, labels, error=exc)
        raise
    finish_run(store, client, result)
    return store.path
