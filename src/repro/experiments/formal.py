"""Run member A's validation comparisons from one immutable in-memory plan."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from repro.artifacts.fingerprints import file_sha256
from repro.artifacts.provenance import environment_snapshot, source_snapshot
from repro.artifacts.store import RunStore
from repro.config import load_config, resolve_path
from repro.datasets.manifests import verify_manifest, verify_manifest_reference
from repro.experiments.baselines import run_baseline
from repro.experiments.evolution import run_evolution_experiment
from repro.serving.lifecycle import managed_transformers_server

DATASETS = ("searchqa", "hotpotqa")
CONDITIONS = ("no_skill", "initial_skill", "original", "selected_val")


def load_formal_configs(path: str | Path, overrides: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Require the agreed course sizes and reserve test inference for a later phase."""
    configs = {
        name: load_config(path, overrides, presets=["course", f"{name}.course"]) for name in DATASETS
    }
    common = configs[DATASETS[0]]["model"]
    for name, config in configs.items():
        if config["model"] != common or common["backend"] != "transformers":
            raise ValueError("Formal comparisons require one shared Transformers model")
        if config["experiment"]["profile"] != "course" or config["experiment"]["allow_test"]:
            raise ValueError("Formal development comparisons require course profile and test disabled")
        if config["data"]["dataset"] != name or not config["data"].get("manifest"):
            raise ValueError("Formal data require the matching recorded dataset identity")
        manifest = verify_manifest(resolve_path(config["data"]["split_dir"]))
        verify_manifest_reference(manifest, resolve_path(config["data"]["manifest"]))
        sizes = {split: item["count"] for split, item in manifest["splits"].items()}
        if manifest["dataset"] != name or manifest.get("synthetic") or sizes != {
            "train": 100, "val": 30, "test": 50,
        }:
            raise ValueError("Formal course data must be real, fixed 100/30/50 partitions")
        evolution = config["evolution"]
        if (
            evolution["generations"] != 3
            or evolution["population_size"] != 2
            or evolution["new_candidates_per_round"] != 2
            or evolution["train_eval_num"] != 15
            or evolution["reflection_failure_k"] != 3
            or evolution["selector"] != "original"
        ):
            raise ValueError("Formal member A comparisons require the agreed three-round original protocol")
    return configs


def run_formal(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> Path:
    """Execute six comparisons and two repeated selected-skill validation evaluations."""
    configs = load_formal_configs(path, overrides)
    report = RunStore(resolve_path(configs[DATASETS[0]]["experiment"]["output_root"]) / "formal")
    report.save_snapshot(configs, environment_snapshot(offline=False), source_snapshot())
    report.save_json("scope.json", {
        "purpose": "member_a_formal_development_comparison", "research_results": True,
        "split": "val", "test_examples_sent_to_model": False,
        "datasets": list(DATASETS), "conditions": list(CONDITIONS), "formal_condition_count": 6,
        "selected_val": "Repeated evaluation of selected skill at the same seed; not a new seed replicate",
        "selection_policy": "Upstream validation selection; no manual replacement after inspecting results",
    })
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    stopped = False
    error = None
    print(f"Formal report: {report.path}", flush=True)
    try:
        with managed_transformers_server(configs[DATASETS[0]]["model"], report.path / "server") as command:
            report.save_json("server_command.json", {"argv": command})
            for dataset in DATASETS:
                best_skill: Path | None = None
                for condition in CONDITIONS:
                    if condition == "selected_val" and best_skill is None:
                        result = {
                            "dataset": dataset, "condition": condition, "status": "skipped",
                            "error": "Original evolution did not produce a completed final skill",
                        }
                        results.append(result)
                        report.record_event("formal_run_finished", **result)
                        continue
                    config = deepcopy(configs[dataset])
                    config["experiment"].update(
                        condition="original" if condition == "selected_val" else condition,
                        output_root=str(report.path / "runs" / dataset / condition),
                    )
                    report.record_event("formal_run_started", dataset=dataset, condition=condition)
                    print(f"Starting {dataset}/{condition}", flush=True)
                    try:
                        if condition == "original":
                            run = run_evolution_experiment(config)
                        else:
                            run = run_baseline(
                                config, split="val",
                                saved_skill=best_skill if condition == "selected_val" else None,
                            )
                        summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
                        result = {
                            "dataset": dataset, "condition": condition, "path": str(run),
                            "status": summary["status"], "summary": summary,
                        }
                        if condition == "original" and summary["status"] == "completed":
                            candidate = run / "upstream/best_skill.md"
                            if not candidate.is_file():
                                raise RuntimeError("Evolution did not save its selected skill")
                            best_skill = candidate
                            report.save_json(f"selection/{dataset}.json", {
                                "path": str(candidate), "sha256": file_sha256(candidate),
                                "selection_split": "val", "test_examples_used": False,
                            })
                    except Exception as exc:
                        if condition == "original":
                            best_skill = None
                        candidates = sorted(
                            Path(config["experiment"]["output_root"]).glob("*/run_summary.json")
                        )
                        result = {
                            "dataset": dataset, "condition": condition, "status": "failed",
                            "path": str(candidates[-1].parent) if candidates else None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    results.append(result)
                    report.record_event("formal_run_finished", **result)
                    print(
                        f"Finished {dataset}/{condition}: {result['status']} {result.get('path')}", flush=True
                    )
        stopped = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        complete = (
            stopped and len(results) == len(DATASETS) * len(CONDITIONS)
            and all(item["status"] == "completed" for item in results)
        )
        report.save_json("summary.json", {
            "status": "completed" if complete else "incomplete", "runs": results,
            "purpose": "member_a_formal_development_comparison", "research_results": True,
            "split": "val", "test_examples_sent_to_model": False,
            "temporary_server_stopped": stopped, "elapsed_s": time.perf_counter() - started, "error": error,
        })
        print(f"Formal report: {report.path} ({'completed' if complete else 'incomplete'})", flush=True)
    return report.path
