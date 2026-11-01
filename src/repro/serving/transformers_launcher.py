"""Build a local Transformers server command from the shared model settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from repro.config import PROJECT_ROOT, load_config, resolve_path
from repro.diagnostics import validate_model_directory


def build_command(settings: dict[str, Any]) -> list[str]:
    if settings["backend"] != "transformers":
        raise ValueError("Transformers serving requires model.backend=transformers")
    model_path = resolve_path(settings["local_path"])
    complete, detail = validate_model_directory(model_path)
    if not complete:
        raise ValueError(f"Incomplete local model: {detail}")
    url = urlsplit(settings["base_url"])
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("The local Transformers server requires an HTTP loopback model.base_url")
    if url.username or url.password or url.path.rstrip("/") != "/v1" or url.query or url.fragment:
        raise ValueError("model.base_url must end in /v1 without credentials, query or fragment")
    command = [
        sys.executable, "-u", "-m", "repro.serving.app",
        "--model-path", str(model_path), "--model-name", settings["name"],
        "--max-model-len", str(settings["max_context_tokens"]),
        "--host", url.hostname, "--port", str(url.port or 80),
    ]
    if settings["enable_thinking"]:
        command.append("--enable-thinking")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--enable-thinking", action="store_true", default=None)
    args = parser.parse_args()
    settings = load_config(args.config)["model"]
    for argument, setting in (
        (args.model_path, "local_path"), (args.model_name, "name"),
        (args.max_model_len, "max_context_tokens"), (args.enable_thinking, "enable_thinking"),
    ):
        if argument is not None:
            settings[setting] = str(argument) if isinstance(argument, Path) else argument
    if args.port is not None:
        url = urlsplit(settings["base_url"])
        if not 1 <= args.port <= 65535:
            raise ValueError("Server port must be between 1 and 65535")
        settings["base_url"] = url._replace(netloc=f"{url.hostname}:{args.port}").geturl()
    return subprocess.call(build_command(settings), cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
