"""Manage only the temporary local server owned by a verification run."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from repro.serving.llama_launcher import build_command


@contextmanager
def managed_llama_server(settings: dict[str, Any], log_dir: Path) -> Iterator[list[str]]:
    """Refuse occupied ports; start hidden; always stop this process on exit."""
    with _managed_server(settings, log_dir, build_command(settings), startup_timeout_s=180) as command:
        yield command


@contextmanager
def managed_transformers_server(settings: dict[str, Any], log_dir: Path) -> Iterator[list[str]]:
    """Load existing local weights for one check and release this server afterwards."""
    from repro.serving.transformers_launcher import build_command as transformers_command

    with _managed_server(settings, log_dir, transformers_command(settings), startup_timeout_s=300) as command:
        yield command


@contextmanager
def _managed_server(
    settings: dict[str, Any], log_dir: Path, command: list[str], *, startup_timeout_s: float,
) -> Iterator[list[str]]:
    log_dir.mkdir(parents=True, exist_ok=False)
    url = urlsplit(settings["base_url"])
    health_url = f"{url.scheme}://{url.netloc}/health"
    with httpx.Client(timeout=2, trust_env=False) as http:
        try:
            http.get(health_url)
        except httpx.TransportError:
            pass
        else:
            raise RuntimeError("Service port is occupied; the existing process was left untouched")
    with (log_dir / "stdout.log").open("x", encoding="utf-8") as stdout:
        with (log_dir / "stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                print(f"Loading local model; server PID={process.pid}; logs={log_dir}", flush=True)
                deadline = time.monotonic() + startup_timeout_s
                with httpx.Client(timeout=2, trust_env=False) as http:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"Model server exited ({process.returncode}); inspect {log_dir}"
                            )
                        try:
                            if http.get(health_url).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Local model loading exceeded {startup_timeout_s} seconds")
                        time.sleep(1)
                yield command
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
