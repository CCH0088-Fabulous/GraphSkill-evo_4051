"""Inspect, install the runtime, or serve an existing local GGUF model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repro.config import load_config, resolve_path
from repro.serving.gguf import inspect_gguf, validate_gguf_file
from repro.serving.llama_install import install_runtime
from repro.serving.llama_launcher import main as serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read local GGUF metadata without loading the model")
    inspect.add_argument("--config", default="configs/local.gguf.yaml")
    inspect.add_argument("--model-path", type=Path)
    inspect.add_argument("--sha256", action="store_true", help="Also hash the model file")
    commands.add_parser("install", help="Install the locked Windows CUDA binaries, never model weights")
    server = commands.add_parser("serve", help="Start the local OpenAI-compatible llama.cpp service")
    server.add_argument("--config", default="configs/local.gguf.yaml")
    server.add_argument("--model-path", type=Path)
    server.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            print(json.dumps(install_runtime(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect":
            settings = load_config(args.config)["model"]
            if settings["backend"] != "llama_cpp":
                raise ValueError("GGUF inspection requires model.backend=llama_cpp")
            path = resolve_path(args.model_path or settings["local_path"])
            metadata = inspect_gguf(path, calculate_sha256=args.sha256)
            ready, detail = validate_gguf_file(path)
            print(json.dumps({"ready": ready, "detail": detail, **metadata}, ensure_ascii=False, indent=2))
            return 0 if ready else 1
        server_args = ["--config", args.config]
        for name in ("model_path", "port"):
            value = getattr(args, name)
            if value is not None:
                server_args.extend(["--" + name.replace("_", "-"), str(value)])
        return serve(server_args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
