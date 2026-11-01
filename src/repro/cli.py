"""CLI argument parsing and dispatch only; work lives in dedicated modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from repro.config import CONDITIONS, PROJECT_ROOT, load_config, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphSkillEvo reproduction infrastructure")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect installed infrastructure without downloading weights")
    doctor.add_argument("--config", default="configs/searchqa.pilot.yaml")
    doctor.add_argument("--preset", action="append", default=[], help="Apply a named config preset")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--endpoint", action="store_true")
    validate = sub.add_parser("validate-data", help="Verify frozen data hashes, IDs, and partition isolation")
    validate.add_argument("--config", required=True)
    validate.add_argument("--preset", action="append", default=[], help="Apply a named config preset")
    show = sub.add_parser("show-config", help="Print the effective configuration without running the model")
    show.add_argument("--config", default="config.yaml")
    show.add_argument("--preset", action="append", default=[], help="Apply a named config preset")
    show.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    freeze = sub.add_parser("freeze", help="Publish an immutable protocol after two-round acceptance")
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--acceptance-run", type=Path, required=True)
    freeze.add_argument("--seeds", type=int, nargs="+", default=[42])
    check = sub.add_parser("verify-freeze", help="Verify protocol files and effective formal settings")
    check.add_argument("--config", required=True)
    check.add_argument("--preset", action="append", default=[], help="Apply a named config preset")
    check.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    acceptance = sub.add_parser("audit-acceptance", help="Audit two-round pilot evidence without inference")
    acceptance.add_argument("--run", type=Path, required=True)
    for name in ("run", "evaluate"):
        experiment = sub.add_parser(
            name, help="Run an experiment" if name == "run" else "Evaluate saved skill"
        )
        experiment.add_argument("--config", required=True)
        experiment.add_argument("--preset", action="append", default=[], help="Apply a named config preset")
        experiment.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
        experiment.add_argument("--condition", choices=CONDITIONS)
        experiment.add_argument("--split", choices=("train", "val", "test"), default="val")
        if name == "evaluate":
            experiment.add_argument("--skill", type=Path, required=True)
    smoke = sub.add_parser("smoke", help="Synthetic offline full loop; produces no research results")
    smoke.add_argument("--dataset", choices=("searchqa", "hotpotqa"), default="searchqa")
    summary = sub.add_parser("summarize", help="Recompute statistics from saved predictions")
    summary.add_argument("--run", type=Path, required=True)
    render = sub.add_parser("render-skill", help="Export a saved skill's internal flowchart")
    render.add_argument("--skill", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "audit-acceptance":
        from repro.artifacts.acceptance import audit_acceptance

        audit = audit_acceptance(args.run)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["status"] == "passed" else 1
    if args.command == "freeze":
        from repro.freezing.publication import publish_freeze

        print(publish_freeze(args.version, args.acceptance_run, seeds=args.seeds))
        return 0
    if args.command == "verify-freeze":
        from repro.freezing.verification import validate_frozen_config

        frozen = validate_frozen_config(load_config(args.config, args.set, presets=args.preset))
        if frozen is None:
            raise ValueError("This configuration has no frozen protocol reference")
        print(json.dumps(frozen, indent=2))
        return 0
    if args.command == "show-config":
        config = load_config(args.config, args.set, presets=args.preset)
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.command == "doctor":
        from repro.diagnostics import collect_diagnostics, doctor_text

        report = collect_diagnostics(
            PROJECT_ROOT, load_config(args.config, presets=args.preset), check_endpoint=args.endpoint
        )
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else doctor_text(report))
        return 0 if report["infrastructure_ready"] else 1
    if args.command == "validate-data":
        from repro.datasets.manifests import verify_manifest, verify_manifest_reference

        config = load_config(args.config, presets=args.preset)
        report = verify_manifest(resolve_path(config["data"]["split_dir"]))
        if report["dataset"] != config["data"]["dataset"]:
            raise ValueError("Configured dataset does not match the dataset manifest")
        if config["data"].get("manifest"):
            verify_manifest_reference(report, resolve_path(config["data"]["manifest"]))
        print(
            json.dumps(
                {
                    "dataset": report["dataset"],
                    "synthetic": report.get("synthetic", False),
                    "counts": {key: item["count"] for key, item in report["splits"].items()},
                },
                indent=2,
            )
        )
        return 0
    if args.command == "summarize":
        from repro.evaluation.summary import summarize_run

        print(json.dumps(summarize_run(args.run), ensure_ascii=False, indent=2))
        return 0
    if args.command == "render-skill":
        from repro.visualization.skill_graph import skill_to_mermaid

        content = skill_to_mermaid(args.skill.read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
        print(args.output.resolve())
        return 0
    if args.command == "smoke":
        return _smoke(args.dataset)
    from repro.experiments.baselines import run_baseline
    from repro.experiments.evolution import run_evolution_experiment

    overrides = list(args.set)
    if args.condition:
        overrides.append(f"experiment.condition={args.condition}")
    config = load_config(args.config, overrides, presets=args.preset)
    if args.command == "evaluate":
        path = run_baseline(config, split=args.split, saved_skill=args.skill.resolve())
    elif config["experiment"]["condition"] in {"no_skill", "initial_skill"}:
        path = run_baseline(config, split=args.split)
    else:
        path = run_evolution_experiment(config)
    print(path)
    return 0


def _smoke(dataset: str) -> int:
    from repro.experiments.baselines import run_baseline
    from repro.experiments.evolution import run_evolution_experiment

    paths = []
    for condition in ("no_skill", "initial_skill", "original"):
        config = load_config(f"configs/offline.{dataset}.yaml", [f"experiment.condition={condition}"])
        paths.append(
            run_evolution_experiment(config, offline=True)
            if condition == "original"
            else run_baseline(config, offline=True)
        )
    print(
        json.dumps(
            {"synthetic": True, "research_results": False, "runs": [str(path) for path in paths]}, indent=2
        )
    )
    return 0
