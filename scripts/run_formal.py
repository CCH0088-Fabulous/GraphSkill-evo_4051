"""Parse and dispatch member A's formal validation comparisons."""

import argparse
import json

from repro.experiments.formal import run_formal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()
    report = run_formal(args.config, args.set)
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
