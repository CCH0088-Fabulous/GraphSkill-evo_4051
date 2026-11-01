"""Parse and dispatch a single local Qwen3.5 training-example check."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from repro.config import PROJECT_ROOT, load_config
from repro.experiments.example import run_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/searchqa.pilot.yaml")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = load_config(args.config, [
        "experiment.condition=initial_skill", "experiment.allow_test=false",
        "experiment.output_root=outputs/qwen35-example",
        f"model.max_completion_tokens={args.max_tokens}", "model.max_retries=0",
        "budget.max_calls=1", f"budget.max_completion_tokens={args.max_tokens}",
    ])
    path = run_example(config)
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"path": str(path), **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
