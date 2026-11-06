"""Evaluate three frozen member-A conditions on a fixed SearchQA or HotpotQA test split."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from repro.artifacts.fingerprints import file_sha256
from repro.artifacts.model_identity import capture_model_identity
from repro.artifacts.provenance import environment_snapshot, source_snapshot
from repro.artifacts.store import RunStore
from repro.config import resolve_path
from repro.datasets.manifests import verify_manifest, verify_manifest_reference
from repro.experiments.baselines import run_baseline
from repro.serving.lifecycle import managed_transformers_server

SEEDS = (0, 42, 127)
CONDITIONS = ("no_skill", "initial_skill", "original")
_DEVELOPMENT_CONDITIONS = (*CONDITIONS, "selected_val")
_SIZES = {"train": 300, "val": 150, "test": 1000}
_DATASET_SIZES = {"searchqa": {"train": 100, "val": 30, "test": 50}, "hotpotqa": _SIZES}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(content: bytes) -> str:
    return content.decode("utf-8").replace("\r\n", "\n")


def _run_path(batch: Path, entry: dict[str, Any], dataset: str) -> Path:
    name = Path(entry["path"].replace("\\", "/")).name
    return batch / "runs" / f"seed-{entry['seed']}" / dataset / entry["condition"] / name


def load_heldout_plan(source_batch: str | Path) -> dict[str, Any]:
    """Require completed validation-only evidence and exact skills/data/model before any request."""
    batch = resolve_path(source_batch)
    summary, scope, plan = (_read(batch / name) for name in ("summary.json", "scope.json", "plan.json"))
    if (
        summary.get("status") != "completed" or summary.get("error")
        or summary.get("experiment_seeds") != list(SEEDS)
        or plan.get("seeds") != list(SEEDS)
        or scope.get("split") != "val" or scope.get("test_examples_sent_to_model") is not False
    ):
        raise ValueError("Held-out evaluation requires the completed validation-only three-seed batch")
    datasets = set(plan["configs_by_seed"][str(SEEDS[0])])
    if len(datasets) != 1 or not datasets <= _DATASET_SIZES.keys():
        raise ValueError("Held-out evaluation requires one supported dataset per source batch")
    dataset = next(iter(datasets))
    sizes = _DATASET_SIZES[dataset]
    entries = {(row["seed"], row["condition"]): row for row in summary["runs"]}
    expected = {(seed, condition) for seed in SEEDS for condition in _DEVELOPMENT_CONDITIONS}
    if len(summary["runs"]) != 12 or set(entries) != expected:
        raise ValueError("Source batch must contain all 12 declared development subruns")
    # Code is not pinned by checksum: skills, data, model and plan identity are verified instead.
    source = _read(batch / "source.json")
    configs, initial, selected, identities, manifests = {}, None, {}, [], []
    evidence = {name: file_sha256(batch / name) for name in (
        "summary.json", "scope.json", "plan.json", "source.json"
    )}
    for seed in SEEDS:
        if set(plan["configs_by_seed"][str(seed)]) != datasets:
            raise ValueError("All source seeds must use the same single dataset")
        config = deepcopy(plan["configs_by_seed"][str(seed)][dataset])
        if (
            config["data"]["dataset"] != dataset or config["model"]["backend"] != "transformers"
            or config["experiment"]["allow_test"] or config["experiment"]["seed"] != seed
            or not config["data"].get("manifest")
        ):
            raise ValueError("Source configuration must use the declared real dataset with test disabled")
        configs[str(seed)] = config
        run_dirs = {}
        for condition in _DEVELOPMENT_CONDITIONS:
            entry = entries[seed, condition]
            if entry.get("dataset", dataset) != dataset:
                raise ValueError("Development entry dataset differs from its frozen plan")
            run = _run_path(batch, entry, dataset)
            run_dirs[condition] = run
            saved = _read(run / "run_summary.json")
            if entry["status"] != "completed" or saved["status"] != "completed" or saved != entry["summary"]:
                raise ValueError("Every development subrun must be completed and match its recorded summary")
            actual = _read(run / "config.json")
            expected_config = deepcopy(config)
            expected_config["experiment"].update(
                condition="original" if condition == "selected_val" else condition,
                output_root=actual["experiment"]["output_root"],
            )
            if actual != expected_config:
                raise ValueError("Development run configuration differs from its frozen plan")
            identities.append(_read(run / "model_identity.json"))
            manifests.append(_read(run / "data_manifest.json"))
            if _read(run / "source.json") != source:
                raise ValueError("Development runs do not share one recorded source revision")
            for name in ("run_summary.json", "config.json", "model_identity.json", "data_manifest.json"):
                evidence[(run / name).relative_to(batch).as_posix()] = file_sha256(run / name)
        original = _read(run_dirs["original"] / "run_summary.json")
        repeated = _read(run_dirs["selected_val"] / "run_summary.json")
        if repeated.get("split") != "val" or repeated.get("n") != sizes["val"]:
            raise ValueError("Selected skill must have a complete validation-only acceptance run")
        if repeated["em"] != original["best"]["val_hard"] or repeated["f1"] != original["best"]["val_soft"]:
            raise ValueError("Repeated validation does not match the automatically selected skill")
        initial_path = run_dirs["initial_skill"] / "skills/evaluated.md"
        records = [json.loads(line) for line in (run_dirs["initial_skill"] / "skills.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()]
        record = next((row for row in records if row["skill_id"] == "evaluated"), None)
        if record is None or record["sha256"] != file_sha256(initial_path):
            raise ValueError("Archived initial skill SHA-256 differs from its development record")
        content = initial_path.read_bytes()
        if initial is None:
            initial = {"content": content, "sha256": file_sha256(initial_path),
                       "source": initial_path.relative_to(batch).as_posix()}
        elif content != initial["content"]:
            raise ValueError("Initial skill differs across development seeds")
        selection_path = batch / f"selection/seed-{seed}/{dataset}.json"
        selection = _read(selection_path)
        skill = run_dirs["original"] / "upstream/best_skill.md"
        suffix = skill.relative_to(batch).as_posix()
        if (
            not selection["path"].replace("\\", "/").endswith(suffix)
            or selection.get("selection_split") != "val" or selection.get("test_examples_used") is not False
            or file_sha256(skill) != selection["sha256"]
            or _text(skill.read_bytes()) != (run_dirs["selected_val"] / "skills/evaluated.md").read_text(
                encoding="utf-8"
            )
        ):
            raise ValueError("Selected skill SHA-256 or repeated-evaluation content differs from the archive")
        selected[str(seed)] = {"content": skill.read_bytes(), "sha256": selection["sha256"],
                               "source": suffix, "skill_id": original["best"]["id"]}
        evidence[selection_path.relative_to(batch).as_posix()] = file_sha256(selection_path)
        evidence[suffix] = selection["sha256"]
        evidence[initial_path.relative_to(batch).as_posix()] = file_sha256(initial_path)
    if (
        any(value != identities[0] for value in identities)
        or any(value != manifests[0] for value in manifests)
    ):
        raise ValueError("Source runs must share exactly the same model identity and data manifest")
    shared = configs[str(SEEDS[0])]
    if any(
        config["model"] != shared["model"] or config["data"] != shared["data"]
        for config in configs.values()
    ):
        raise ValueError("All source seeds must share model and data settings")
    manifest = verify_manifest(resolve_path(shared["data"]["split_dir"]))
    verify_manifest_reference(manifest, resolve_path(shared["data"]["manifest"]))
    if (
        manifest != manifests[0] or manifest.get("synthetic") or manifest.get("dataset") != dataset
        or {key: value["count"] for key, value in manifest["splits"].items()} != sizes
    ):
        raise ValueError(f"Held-out data must match the source's fixed real {dataset} partitions: {sizes}")
    current_identity = capture_model_identity(shared["model"], synthetic=False)
    for key in ("fingerprint_sha256", "precision", "attention_implementation", "device"):
        if current_identity.get(key) != identities[0].get(key):
            raise ValueError("Current model fingerprint or inference runtime differs from development")
    if not current_identity.get("fingerprint_sha256"):
        raise ValueError("A verified local model fingerprint is required")
    return {"source_batch": batch, "dataset": dataset, "configs": configs,
            "initial": initial, "selected": selected,
            "data_manifest": manifest, "model_identity": current_identity, "source_evidence": evidence}


def run_heldout(source_batch: str | Path) -> Path:
    """Freeze all evaluation inputs before launching one service; never invoke optimization."""
    frozen = load_heldout_plan(source_batch)
    source = frozen["source_batch"]
    first = frozen["configs"][str(SEEDS[0])]
    report = RunStore(resolve_path(first["experiment"]["output_root"]) / f"heldout-{frozen['dataset']}")
    report.save_snapshot(frozen["configs"], environment_snapshot(offline=False), source_snapshot())
    report.save_json("source_evidence.json", frozen["source_evidence"])
    report.save_json("model_identity.json", frozen["model_identity"])
    data_root = report.path / "frozen/data"
    data_root.mkdir(parents=True)
    original_data = resolve_path(first["data"]["split_dir"])
    files = {}
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "split_manifest.json"):
        target = data_root / name
        with target.open("xb") as stream:
            stream.write((original_data / name).read_bytes())
        files[target.relative_to(report.path).as_posix()] = file_sha256(target)
    if verify_manifest(data_root) != frozen["data_manifest"]:
        raise ValueError("Data changed while the evaluation snapshot was being copied")
    snapshots = {
        "initial": frozen["initial"],
        **{f"seed-{seed}": frozen["selected"][str(seed)] for seed in SEEDS},
    }
    skill_paths = {}
    for name, entry in snapshots.items():
        target = report.path / "frozen/skills" / f"{name}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(entry["content"])
        if file_sha256(target) != entry["sha256"]:
            raise ValueError("Frozen skill differs from the selected source")
        skill_paths[name] = target
        files[target.relative_to(report.path).as_posix()] = entry["sha256"]
    jobs = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            config = deepcopy(frozen["configs"][str(seed)])
            config["experiment"].update(
                allow_test=True, condition=condition, initial_skill=str(skill_paths["initial"]),
                output_root=str(report.path / "runs" / f"seed-{seed}" / condition),
            )
            config["data"].update(split_dir=str(data_root), manifest=str(data_root / "split_manifest.json"))
            jobs.append({"seed": seed, "condition": condition, "config": config,
                         "skill": None if condition == "no_skill" else str(skill_paths[
                             "initial" if condition == "initial_skill" else f"seed-{seed}"
                         ])})
    scope = {
        "purpose": "member_a_frozen_heldout_evaluation", "source_batch": str(source),
        "dataset": frozen["dataset"], "split": "test",
        "n_per_run": frozen["data_manifest"]["splits"]["test"]["count"],
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "optimization_allowed": False, "test_feedback_to_method_development_allowed": False,
        "member_b_improved_and_ablation_included": False,
        "note": "Only frozen A conditions; no skill reselection or inference-setting tuning after test",
    }
    report.save_json("scope.json", scope)
    report.save_json("freeze.json", {"source_batch": str(source), "files_sha256": files,
        "skills": {
            name: {k: v for k, v in entry.items() if k != "content"} for name, entry in snapshots.items()
        },
        "model_fingerprint_sha256": frozen["model_identity"]["fingerprint_sha256"],
        "test_ids": frozen["data_manifest"]["splits"]["test"]["ids"], **scope})
    report.save_json("plan.json", {"jobs": jobs})
    print(f"Held-out report: {report.path}", flush=True)
    return _execute(report, jobs, files, scope, frozen, done={}, attempt=0)


def _unused(directory: Path, stem: str, suffix: str) -> Path:
    number = 1
    while (directory / f"{stem}-{number}{suffix}").exists():
        number += 1
    return directory / f"{stem}-{number}{suffix}"


def resume_heldout(held_out_batch: str | Path) -> Path:
    """Continue an interrupted held-out batch; keep finished subruns and rerun the rest in new dirs."""
    path = resolve_path(held_out_batch)
    freeze, plan, scope = (_read(path / name) for name in ("freeze.json", "plan.json", "scope.json"))
    if scope.get("purpose") != "member_a_frozen_heldout_evaluation" or scope.get("optimization_allowed"):
        raise ValueError("Not a frozen member-A held-out batch")
    frozen = load_heldout_plan(freeze["source_batch"])
    if _read(path / "source_evidence.json") != frozen["source_evidence"]:
        raise ValueError("Development evidence differs from the evidence frozen with this batch")
    if (
        frozen["model_identity"].get("fingerprint_sha256") != freeze["model_fingerprint_sha256"]
        or frozen["data_manifest"]["splits"]["test"]["ids"] != freeze["test_ids"]
    ):
        raise ValueError("Current model or test IDs differ from the frozen batch")
    files = freeze["files_sha256"]
    for relative, checksum in files.items():
        if file_sha256(path / relative) != checksum:
            raise ValueError("Frozen evaluation input changed since the batch was created")
    jobs = plan["jobs"]
    done: dict[tuple[int, str], dict[str, Any]] = {}
    events = path / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("event") != "heldout_run_finished":
            continue
        summary = row.get("summary") or {}
        if (
            row.get("status") in {"completed", "completed_with_errors"}
            and summary.get("n") == scope["n_per_run"] and summary.get("split") == "test"
        ):
            done[row["seed"], row["condition"]] = {
                key: value for key, value in row.items() if key not in {"event", "timestamp"}
            }
    attempt = sum(1 for row in rows if row.get("event") == "heldout_resumed") + 1
    if (path / "summary.json").exists():
        (path / "summary.json").rename(_unused(path, "summary.attempt", ".json"))
    report = RunStore.open_existing(path)
    report.record_event("heldout_resumed", attempt=attempt, finished=len(done), pending=len(jobs) - len(done))
    print(f"Resuming held-out report: {path}; {len(done)}/{len(jobs)} subruns finished", flush=True)
    return _execute(report, jobs, files, scope, frozen, done=done, attempt=attempt)


def _execute(
    report: RunStore, jobs: list[dict[str, Any]], files: dict[str, str], scope: dict[str, Any],
    frozen: dict[str, Any], *, done: dict[tuple[int, str], dict[str, Any]], attempt: int,
) -> Path:
    first = frozen["configs"][str(SEEDS[0])]
    results: list[dict[str, Any]] = []
    stopped, error = False, None
    started = time.perf_counter()
    suffix = "" if attempt == 0 else f"-resume-{attempt}"
    try:
        with managed_transformers_server(first["model"], report.path / f"server{suffix}") as command:
            report.save_json(f"server_command{suffix}.json", {"argv": command})
            for index, job in enumerate(jobs):
                if (job["seed"], job["condition"]) in done:
                    results.append(done[job["seed"], job["condition"]])
                    continue
                for relative, checksum in files.items():
                    if file_sha256(report.path / relative) != checksum:
                        raise ValueError("Frozen evaluation input changed after the plan was recorded")
                print(f"Starting test seed={job['seed']}/{job['condition']}", flush=True)
                report.record_event("heldout_run_started", seed=job["seed"], condition=job["condition"])
                result = {"seed": job["seed"], "condition": job["condition"]}
                try:
                    path = run_baseline(deepcopy(job["config"]), split="test",
                                        saved_skill=Path(job["skill"]) if job["skill"] else None)
                    summary = _read(path / "run_summary.json")
                    result.update(path=str(path), status=summary["status"], summary=summary)
                    if (
                        summary["status"] not in {"completed", "completed_with_errors"}
                        or summary.get("n") != scope["n_per_run"] or summary.get("split") != "test"
                    ):
                        raise RuntimeError("Held-out subrun did not complete all fixed test examples")
                    identity = _read(path / "model_identity.json")
                    if identity.get("fingerprint_sha256") != frozen["model_identity"]["fingerprint_sha256"]:
                        raise RuntimeError("Evaluated model differs from the frozen development model")
                    if _read(path / "data_manifest.json") != frozen["data_manifest"]:
                        raise RuntimeError("Evaluated dataset differs from the frozen test identity")
                except Exception as exc:
                    result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    candidates = sorted(Path(job["config"]["experiment"]["output_root"]).glob(
                        "*/run_summary.json"
                    ))
                    if candidates:
                        result["path"] = str(candidates[-1].parent)
                    error = result["error"]
                results.append(result)
                report.record_event("heldout_run_finished", **result)
                print(f"Finished test seed={job['seed']}/{job['condition']}: {result['status']}", flush=True)
                if error:
                    results.extend({"seed": pending["seed"], "condition": pending["condition"],
                                    "status": "not_run", "reason": "Stopped after an earlier subrun failed"}
                                   for pending in jobs[index + 1:]
                                   if (pending["seed"], pending["condition"]) not in done)
                    break
        stopped = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        complete = stopped and len(results) == len(jobs) and all(
            row["status"] in {"completed", "completed_with_errors"} for row in results
        )
        warnings = any(row["status"] == "completed_with_errors" for row in results)
        status = ("completed_with_errors" if warnings else "completed") if complete else "incomplete"
        report.save_json("summary.json", {"status": status, "resume_attempt": attempt,
            "runs": results, "temporary_server_stopped": stopped, "elapsed_s": time.perf_counter() - started,
            "error": error, **scope})
        print(f"Held-out report: {report.path} ({status})", flush=True)
    return report.path
