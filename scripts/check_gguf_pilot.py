"""Verify one real evolution round per dataset using existing pilot IDs only."""

from __future__ import annotations

import json
import time

from dotenv import load_dotenv

from repro.artifacts.store import RunStore
from repro.config import PROJECT_ROOT, load_config
from repro.diagnostics import collect_diagnostics
from repro.experiments.evolution import run_evolution_experiment
from repro.serving.lifecycle import managed_llama_server


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    shared = load_config(PROJECT_ROOT / "configs/local.gguf.yaml")
    report = RunStore(PROJECT_ROOT / "outputs/gguf-pilot-check")
    summary = {"research_comparison": False, "test_examples_used": False, "runs": []}
    started = time.perf_counter()
    try:
        with managed_llama_server(shared["model"], report.path / "server") as command:
            report.save_json("server_command.json", command)
            for dataset in ("searchqa", "hotpotqa"):
                config = load_config(PROJECT_ROOT / f"configs/{dataset}.pilot.yaml")
                config["model"] = dict(shared["model"])
                if config["experiment"]["allow_test"] or config["evolution"]["generations"] != 1:
                    raise ValueError("Pilot verification requires one evolution round and test disabled")
                print(f"Starting real {dataset} pilot evolution", flush=True)
                path = run_evolution_experiment(config)
                run_summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
                summary["runs"].append(
                    {
                        "dataset": dataset,
                        "path": str(path),
                        "status": run_summary["status"],
                        "usage": run_summary["usage"],
                        "failed_predictions": run_summary["failed_predictions"],
                    }
                )
                print(f"Completed {dataset}: {path}", flush=True)
            report.save_json(
                "doctor_with_service.json",
                collect_diagnostics(
                    PROJECT_ROOT,
                    config,
                    check_endpoint=True,
                ),
            )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary.update(elapsed_s=time.perf_counter() - started, temporary_server_stopped=True)
        report.save_json("summary.json", summary)
        print(f"Verification report: {report.path}", flush=True)


if __name__ == "__main__":
    main()
