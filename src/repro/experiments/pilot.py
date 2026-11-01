"""Run the two frozen pilot datasets with one local model service."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from repro.artifacts.store import RunStore
from repro.config import PROJECT_ROOT, load_config, resolve_path
from repro.datasets.manifests import verify_manifest
from repro.experiments.baselines import run_baseline
from repro.experiments.evolution import run_evolution_experiment
from repro.serving.lifecycle import managed_transformers_server

_DATASETS = ("searchqa", "hotpotqa")
_CONDITIONS = ("no_skill", "initial_skill", "original", "selected_val")


def load_pilot_configs(
    overrides: list[str] | None = None, *, profile: str = "pilot"
) -> dict[str, dict[str, Any]]:
    """Reject mismatched models, data boundaries or accidental test use before starting."""
    if profile not in {"pilot", "acceptance"}:
        raise ValueError("Unknown pilot profile")
    configs = {
        name: load_config(PROJECT_ROOT / f"configs/{name}.{profile}.yaml", overrides or [])
        for name in _DATASETS
    }
    shared_model = configs[_DATASETS[0]]["model"]
    for name, config in configs.items():
        manifest = verify_manifest(resolve_path(config["data"]["split_dir"]))
        if manifest["dataset"] != name or config["data"]["dataset"] != name:
            raise ValueError(f"Dataset mismatch for {name}")
        if manifest.get("synthetic") or manifest["splits"]["test"]["count"]:
            raise ValueError(f"{name} pilot must contain real data and no test examples")
        expected_rounds = 2 if profile == "acceptance" else 1
        if config["experiment"]["allow_test"] or config["evolution"]["generations"] != expected_rounds:
            raise ValueError(
                f"{profile} requires test disabled and exactly {expected_rounds} evolution rounds"
            )
        if config["model"] != shared_model or shared_model["backend"] != "transformers":
            raise ValueError("Both pilots must use the same configured Transformers model")
    return configs


def run_pilots(overrides: list[str] | None = None, *, profile: str = "pilot") -> Path:
    configs = load_pilot_configs(overrides, profile=profile)
    report = RunStore(PROJECT_ROOT / "outputs/pilot-check")
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    server_stopped = False
    error = None
    report.save_json(
        "scope.json",
        {
            "purpose": "real_data_pilot",
            "research_results": False,
            "test_examples_used": False,
            "datasets": list(_DATASETS),
            "conditions": list(_CONDITIONS),
            "profile": profile,
        },
    )
    print(f"Pilot report: {report.path}", flush=True)
    try:
        with managed_transformers_server(configs[_DATASETS[0]]["model"], report.path / "server") as argv:
            report.save_json("server_command.json", {"argv": argv})
            for dataset in _DATASETS:
                best_skill: Path | None = None
                for condition in _CONDITIONS:
                    if condition == "selected_val" and best_skill is None:
                        results.append(
                            {
                                "dataset": dataset,
                                "condition": condition,
                                "status": "skipped",
                                "error": "Original evolution did not produce a final skill",
                            }
                        )
                        continue
                    selected_condition = "original" if condition == "selected_val" else condition
                    output_root = report.path / "runs" / dataset / condition
                    config = load_config(
                        PROJECT_ROOT / f"configs/{dataset}.{profile}.yaml",
                        [
                            *(overrides or []),
                            f"experiment.condition={selected_condition}",
                            f"experiment.output_root={output_root.as_posix()}",
                        ],
                    )
                    report.record_event("pilot_run_started", dataset=dataset, condition=condition)
                    print(f"Starting {dataset}/{condition}", flush=True)
                    try:
                        if condition == "original":
                            path = run_evolution_experiment(config)
                        else:
                            path = run_baseline(
                                config,
                                split="val",
                                saved_skill=best_skill if condition == "selected_val" else None,
                            )
                        summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
                        if condition == "original":
                            candidate = path / "upstream/best_skill.md"
                            if summary["status"] == "completed" and candidate.is_file():
                                best_skill = candidate
                        result = {
                            "dataset": dataset,
                            "condition": condition,
                            "path": str(path),
                            "status": summary["status"],
                            "error": summary.get("error"),
                            "usage": summary["usage"],
                        }
                        if condition == "original" and best_skill is None:
                            result.update(
                                status="failed", error="Evolution did not save a completed final skill"
                            )
                    except Exception as exc:
                        candidates = (
                            sorted(output_root.glob("*/run_summary.json")) if output_root.exists() else []
                        )
                        result = {
                            "dataset": dataset,
                            "condition": condition,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "path": str(candidates[-1].parent) if candidates else None,
                        }
                    results.append(result)
                    report.record_event("pilot_run_finished", **result)
                    print(
                        f"Finished {dataset}/{condition}: {result['status']} {result.get('path')}", flush=True
                    )
        server_stopped = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status = (
            "completed"
            if len(results) == len(_DATASETS) * len(_CONDITIONS)
            and all(result["status"] == "completed" for result in results)
            and server_stopped
            else "incomplete"
        )
        report.save_json(
            "summary.json",
            {
                "status": status,
                "purpose": "real_data_pilot",
                "research_results": False,
                "test_examples_used": False,
                "temporary_server_stopped": server_stopped,
                "elapsed_s": time.perf_counter() - started,
                "runs": results,
                "error": error,
            },
        )
        print(f"Pilot report: {report.path} ({status})", flush=True)
    return report.path
