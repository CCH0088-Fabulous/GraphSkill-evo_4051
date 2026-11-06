"""Dispatch member A's development comparisons or frozen held-out evaluation."""

import argparse
import json

from repro.experiments.formal import DATASETS, run_formal
from repro.experiments.heldout import resume_heldout, run_heldout
from repro.experiments.provisional import run_provisional


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=DATASETS, help="Run course development for one dataset only")
    parser.add_argument(
        "--test-from-batch", help="Evaluate all frozen A conditions from a completed development batch"
    )
    parser.add_argument(
        "--resume-heldout", metavar="BATCH",
        help="Continue an interrupted held-out batch directory; finished subruns are kept",
    )
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument(
        "--provisional-hotpotqa", action="store_true",
        help="Use the provisional stratified 300/150/1000 HotpotQA protocol; never infer on test",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="Distinct experiment seeds for the selected dataset/protocol",
    )
    args = parser.parse_args(argv)
    if args.resume_heldout:
        if (
            args.test_from_batch or args.config or args.set
            or args.seeds is not None or args.provisional_hotpotqa or args.dataset
        ):
            parser.error("--resume-heldout uses the frozen batch plan; no other options are accepted")
        report = resume_heldout(args.resume_heldout)
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        return 0 if summary["status"] == "completed" else 1
    if args.test_from_batch:
        if args.config or args.set or args.seeds is not None or args.provisional_hotpotqa or args.dataset:
            parser.error("--test-from-batch uses the frozen source plan; config/set/seeds are not accepted")
        report = run_heldout(args.test_from_batch)
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        return 0 if summary["status"] == "completed" else 1
    if args.dataset and args.provisional_hotpotqa:
        parser.error("--dataset and --provisional-hotpotqa select different protocols; choose one")
    if args.seeds is not None and not (args.provisional_hotpotqa or args.dataset):
        parser.error("--seeds requires --dataset or --provisional-hotpotqa")
    if args.seeds is not None and any(item.split("=", 1)[0] == "experiment.seed" for item in args.set):
        parser.error("Specify experiment seeds with either --seeds or --set experiment.seed, not both")
    report = (
        run_provisional(args.config or "config.yaml", args.set, seeds=args.seeds)
        if args.provisional_hotpotqa else run_formal(
            args.config or "config.yaml", args.set,
            datasets=(args.dataset,) if args.dataset else DATASETS, seeds=args.seeds,
        )
    )
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
