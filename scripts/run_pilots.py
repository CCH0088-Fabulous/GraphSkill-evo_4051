"""Run both local real-data pilots and print their batch report directory."""

import argparse
import json

from repro.experiments.pilot import run_pilots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--profile", choices=("pilot", "acceptance"), default="pilot")
    args = parser.parse_args(argv)
    path = run_pilots(args.set, profile=args.profile)
    print(path, flush=True)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
