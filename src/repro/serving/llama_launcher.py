"""Build and execute one local GGUF server command with lossless argument quoting."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from repro.config import PROJECT_ROOT, load_config, resolve_path
from repro.serving.llama_runtime import locate_server


def build_command(
    settings: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    model_path: str | Path | None = None,
    server_path: str | Path | None = None,
    port: int | None = None,
) -> list[str]:
    if settings.get("backend", "transformers") != "llama_cpp":
        raise ValueError("GGUF serving requires model.backend=llama_cpp")
    model = resolve_path(model_path or settings["local_path"], root=project_root)
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("Model path must point to an existing local GGUF file")
    parsed = urlsplit(settings["base_url"])
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("The local llama.cpp server requires an HTTP loopback model.base_url")
    if parsed.username or parsed.password:
        raise ValueError("Do not embed credentials in model.base_url")
    selected_port = port if port is not None else parsed.port or 8000
    if not 1 <= selected_port <= 65535:
        raise ValueError("Server port must be between 1 and 65535")
    server, _lock = locate_server(project_root, server_path)
    return [
        str(server),
        "--model",
        str(model),
        "--alias",
        settings["name"],
        "--ctx-size",
        str(settings["max_context_tokens"]),
        "--parallel",
        "1",
        "--predict",
        str(max(settings.get("max_completion_tokens", 2048), 4096)),
        "--n-gpu-layers",
        "all",
        "--host",
        "127.0.0.1",
        "--port",
        str(selected_port),
        "--jinja",
        "--no-context-shift",
        "--reasoning",
        "on" if settings["enable_thinking"] else "off",
        "--reasoning-format",
        "none",
        "--flash-attn",
        "on",
        "--chat-template-kwargs",
        json.dumps({"enable_thinking": settings["enable_thinking"]}),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--server-path", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = load_config(args.config)["model"]
    command = build_command(
        settings,
        model_path=args.model_path,
        server_path=args.server_path,
        port=args.port,
    )
    from repro.serving.gguf import validate_gguf_file

    model_index = command.index("--model") + 1
    valid, message = validate_gguf_file(Path(command[model_index]))
    if not valid:
        raise ValueError(message)
    print(f"Starting pinned local GGUF server: {settings['name']}", flush=True)
    # A sequence avoids PowerShell's legacy JSON/native-command quoting conversions.
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
