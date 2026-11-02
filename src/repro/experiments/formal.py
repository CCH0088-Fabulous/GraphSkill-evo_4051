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
        validate_original_protocol(config)
    return configs


def validate_original_protocol(config: dict[str, Any]) -> None:
    """Keep the original optimizer identical across both dataset protocols."""
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


def run_formal(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> Path:
    """Execute six comparisons and two repeated selected-skill validation evaluations."""
    configs = load_formal_configs(path, overrides)
    return run_comparison_batch(configs)


def _experiment_seeds(configs: dict[str, dict[str, Any]], seeds: list[int] | None) -> list[int]:
    configured = [config["experiment"]["seed"] for config in configs.values()]
    if not configured or len(set(configured)) != 1:
        raise ValueError("Comparison datasets require the same configured experiment seed")
    selected = list(seeds) if seeds is not None else [configured[0]]
    if (
        not selected
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("Experiment seeds must be a nonempty list of distinct nonnegative integers")
    return selected


def run_comparison_batch(
    configs: dict[str, dict[str, Any]],
    *,
    seeds: list[int] | None = None,
    folder: str = "formal",
    purpose: str = "member_a_formal_development_comparison",
    protocol: dict[str, Any] | None = None,
) -> Path:
    """Run an already validated plan using one service and isolated seed/condition outputs."""
    selected_seeds = _experiment_seeds(configs, seeds)
    datasets = tuple(configs)
    shared_model = configs[datasets[0]]["model"]
    if any(config["model"] != shared_model for config in configs.values()):
        raise ValueError("Comparison datasets require one shared model")
    if any(config["experiment"]["allow_test"] for config in configs.values()):
        raise ValueError("Development comparisons require test disabled")
    report = RunStore(resolve_path(configs[datasets[0]]["experiment"]["output_root"]) / folder)
    report.save_snapshot(configs, environment_snapshot(offline=False), source_snapshot())
    plans = {
        str(seed): {
            dataset: {
                **deepcopy(config),
                "experiment": {**deepcopy(config["experiment"]), "seed": seed},
            }
            for dataset, config in configs.items()
        }
        for seed in selected_seeds
    }
    report.save_json("plan.json", {"seeds": selected_seeds, "configs_by_seed": plans})
    report.save_json("scope.json", {
        "purpose": purpose, "research_results": True,
        "split": "val", "test_examples_sent_to_model": False,
        "datasets": list(datasets), "conditions": list(CONDITIONS),
        "experiment_seeds": selected_seeds, "formal_condition_count": 3 * len(datasets) * len(selected_seeds),
        "selected_val": "Repeated evaluation of selected skill at the same seed; not a new seed replicate",
        "selection_policy": "Upstream validation selection; no manual replacement after inspecting results",
        "protocol": protocol or {},
    })
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    stopped = False
    error = None
    print(f"Formal report: {report.path}", flush=True)
    try:
        with managed_transformers_server(shared_model, report.path / "server") as command:
            report.save_json("server_command.json", {"argv": command})
            for seed, dataset in ((seed, dataset) for seed in selected_seeds for dataset in datasets):
                best_skill: Path | None = None
                for condition in CONDITIONS:
                    if condition == "selected_val" and best_skill is None:
                        result = {
                            "dataset": dataset, "seed": seed, "condition": condition, "status": "skipped",
                            "error": "Original evolution did not produce a completed final skill",
                        }
                        results.append(result)
                        report.record_event("formal_run_finished", **result)
                        continue
                    config = deepcopy(plans[str(seed)][dataset])
                    run_root = report.path / "runs"
                    if seeds is not None:
                        run_root /= f"seed-{seed}"
                    config["experiment"].update(
                        condition="original" if condition == "selected_val" else condition,
                        output_root=str(run_root / dataset / condition),
                    )
                    report.record_event("formal_run_started", dataset=dataset, seed=seed, condition=condition)
                    print(f"Starting seed={seed}/{dataset}/{condition}", flush=True)
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
                            "dataset": dataset, "seed": seed, "condition": condition, "path": str(run),
                            "status": summary["status"], "summary": summary,
                        }
                        if condition == "original" and summary["status"] == "completed":
                            candidate = run / "upstream/best_skill.md"
                            if not candidate.is_file():
                                raise RuntimeError("Evolution did not save its selected skill")
                            best_skill = candidate
                            selection = f"selection/{dataset}.json" if seeds is None else (
                                f"selection/seed-{seed}/{dataset}.json"
                            )
                            report.save_json(selection, {
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
                            "dataset": dataset, "seed": seed, "condition": condition, "status": "failed",
                            "path": str(candidates[-1].parent) if candidates else None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    results.append(result)
                    report.record_event("formal_run_finished", **result)
                    print(
                        f"Finished seed={seed}/{dataset}/{condition}: "
                        f"{result['status']} {result.get('path')}",
                        flush=True,
                    )
        stopped = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        complete = (
            stopped and len(results) == len(datasets) * len(CONDITIONS) * len(selected_seeds)
            and all(item["status"] == "completed" for item in results)
        )
        report.save_json("summary.json", {
            "status": "completed" if complete else "incomplete", "runs": results,
            "purpose": purpose, "research_results": True,
            "split": "val", "test_examples_sent_to_model": False,
            "experiment_seeds": selected_seeds, "protocol": protocol or {},
            "temporary_server_stopped": stopped, "elapsed_s": time.perf_counter() - started, "error": error,
        })
        print(f"Formal report: {report.path} ({'completed' if complete else 'incomplete'})", flush=True)
    return report.path
