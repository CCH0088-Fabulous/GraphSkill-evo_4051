"""Read-only environment checks; this module never downloads models or installs software."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _check(name: str, status: str, message: str, *, required: bool = True, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "required": required, **details}


def _decode_output(output: bytes) -> str:
    if b"\x00" in output:
        return output.decode("utf-16-le", errors="replace").strip()
    return output.decode("utf-8", errors="replace").strip()


def run_command(command: Sequence[str], *, timeout_s: float = 15) -> tuple[int, str]:
    """Capture one command without invoking a shell or allowing interactive input."""
    try:
        result = subprocess.run(
            list(command), capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    output = _decode_output(result.stdout)
    error = _decode_output(result.stderr)
    return result.returncode, "\n".join(part for part in (output, error) if part)


def check_python() -> dict[str, Any]:
    supported = sys.version_info >= (3, 11)
    return _check(
        "python", "ok" if supported else "error", platform.python_version(), executable=sys.executable
    )


def check_dependencies() -> dict[str, Any]:
    versions: dict[str, str] = {}
    missing = []
    for package in ("graphskill-repro", "graphskillevo", "httpx", "openai", "PyYAML", "python-dotenv"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    return _check(
        "development_dependencies",
        "missing" if missing else "ok",
        f"Missing: {', '.join(missing)}" if missing else "Development and upstream packages installed",
        versions=versions,
    )


def check_upstream(project_root: Path) -> dict[str, Any]:
    checkout = project_root / "third_party" / "GraphSkillEvo"
    lock_file = project_root / "third_party" / "upstream.lock.json"
    if not lock_file.is_file() or not (checkout / "graphskillevo" / "trainer.py").is_file():
        return _check("upstream", "missing", "Pinned official source or upstream.lock.json is missing")
    try:
        expected = json.loads(lock_file.read_text(encoding="utf-8"))["commit"]
    except (OSError, ValueError, KeyError) as exc:
        return _check("upstream", "error", f"Invalid upstream lock: {exc}")
    status, actual = run_command(
        ["git", "-c", f"safe.directory={checkout.as_posix()}", "-C", str(checkout), "rev-parse", "HEAD"]
    )
    if status:
        return _check("upstream", "error", actual)
    return _check(
        "upstream",
        "ok" if actual == expected else "error",
        "Official checkout matches pinned commit"
        if actual == expected
        else "Official commit does not match lock",
        expected_commit=expected,
        actual_commit=actual,
    )


def check_gpu() -> dict[str, Any]:
    status, output = run_command(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    )
    if status:
        return _check("gpu", "missing", "NVIDIA GPU/driver is unavailable", command_output=output)
    try:
        rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
        devices = [
            {"name": row[0], "memory_mib": int(row[1]), "driver": row[2]} for row in rows if len(row) == 3
        ]
    except (ValueError, IndexError):
        return _check("gpu", "error", "Unexpected nvidia-smi output", command_output=output)
    if not devices:
        return _check("gpu", "missing", "No GPU reported by nvidia-smi")
    return _check("gpu", "ok", "NVIDIA GPU detected", devices=devices)


def check_wsl() -> dict[str, Any]:
    if platform.system() != "Windows":
        return _check("wsl", "ok", "Native Linux does not require WSL", required=False)
    status, output = run_command(["wsl", "--list", "--quiet"])
    distributions = [line.strip() for line in output.splitlines() if line.strip()]
    if status or not distributions:
        return _check(
            "wsl",
            "deferred",
            "Optional WSL is not installed; native Windows inference does not require it",
            required=False,
            command_output=output,
        )
    return _check(
        "wsl", "ok", "Optional WSL distribution detected", required=False, distributions=distributions
    )


def _check_llama_runtime(project_root: Path) -> dict[str, Any]:
    from repro.serving.llama_runtime import locate_server

    try:
        executable, lock = locate_server(project_root)
    except FileNotFoundError as exc:
        return _check("inference_runtime", "missing", f"Pinned llama.cpp runtime is incomplete: {exc}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _check("inference_runtime", "error", f"Invalid llama.cpp runtime pin: {exc}")
    status, version = run_command([str(executable), "--version"])
    if status:
        return _check("inference_runtime", "error", "llama-server could not start", detail=version)
    status, devices = run_command([str(executable), "--list-devices"])
    cuda_visible = status == 0 and "CUDA" in devices
    return _check(
        "inference_runtime",
        "ok" if cuda_visible else "error",
        "Pinned llama.cpp CUDA runtime is available"
        if cuda_visible
        else "llama.cpp did not report a CUDA device",
        backend="llama_cpp",
        release=lock["release"],
        executable=str(executable),
        version=version,
        devices=devices,
    )


def check_inference_runtime(
    project_root: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (config or {}).get("model", {}).get("backend", "transformers") == "llama_cpp":
        return _check_llama_runtime(project_root)
    if platform.system() == "Windows":
        python_path = project_root / ".venv" / "Scripts" / "python.exe"
    elif platform.system() == "Linux":
        python_path = project_root / ".venv-linux" / "bin" / "python"
    else:
        return _check(
            "inference_runtime", "missing", "The NVIDIA inference workflow requires Windows or Linux"
        )
    command = [str(python_path)]
    code = (
        "import json,torch,transformers,fastapi,uvicorn,accelerate; "
        "from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration; "
        "gpu_ok=torch.cuda.is_available(); "
        "x=torch.ones((2,2),device='cuda',dtype=torch.bfloat16) if gpu_ok else None; "
        "gpu_result=float((x@x).sum().item()) if gpu_ok else None; "
        "print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,"
        "'fastapi':fastapi.__version__,'uvicorn':uvicorn.__version__,"
        "'cuda':torch.version.cuda,'gpu_available':gpu_ok,'bf16_matmul_result':gpu_result}))"
    )
    status, output = run_command([*command, "-c", code], timeout_s=45)
    if status:
        return _check(
            "inference_runtime",
            "missing",
            "Native server dependencies or CUDA verification failed",
            detail=output,
        )
    try:
        versions = json.loads(output.splitlines()[-1])
    except (ValueError, IndexError):
        return _check(
            "inference_runtime", "error", "Could not parse inference environment response", detail=output
        )
    return _check(
        "inference_runtime",
        "ok" if versions.get("gpu_available") else "error",
        "Transformers server dependencies installed; CUDA visible"
        if versions.get("gpu_available")
        else "CUDA is not visible",
        versions=versions,
    )


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def check_data(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    from repro.datasets.manifests import COURSE_LIMITS, verify_manifest, verify_manifest_reference

    data = config.get("data", {})
    split_dir = data.get("split_dir")
    if not split_dir:
        return _check("data", "missing", "No dataset split directory configured")
    resolved = _resolve(project_root, split_dir)
    expected_manifest = data.get("manifest")
    try:
        manifest = verify_manifest(resolved)
        if manifest["dataset"] != data.get("dataset"):
            raise ValueError("Manifest dataset does not match data.dataset")
        if expected_manifest:
            verify_manifest_reference(manifest, _resolve(project_root, expected_manifest))
        profile = config.get("experiment", {}).get("profile")
        if profile == "course":
            if manifest.get("synthetic"):
                raise ValueError("Synthetic data cannot be used with the course profile")
            if any(manifest["splits"][name]["count"] > limit for name, limit in COURSE_LIMITS.items()):
                raise ValueError("Fixed subset exceeds course profile limits")
    except FileNotFoundError as exc:
        return _check("data", "missing", f"Frozen dataset is incomplete: {exc}", split_dir=str(resolved))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _check("data", "error", f"Dataset verification failed: {exc}", split_dir=str(resolved))
    return _check(
        "data",
        "ok",
        "Frozen manifest, checksums, IDs and split boundaries verified",
        dataset=data.get("dataset"),
        split_dir=str(resolved),
        counts={name: entry["count"] for name, entry in manifest["splits"].items()},
        synthetic=manifest.get("synthetic", False),
    )


def validate_model_directory(model_path: Path) -> tuple[bool, str]:
    """Check local Hugging Face model payload completeness without contacting the hub."""
    model_path = Path(model_path)
    if not model_path.is_dir():
        return False, "Model path is not an existing local directory"
    for name in ("config.json", "tokenizer_config.json"):
        path = model_path / name
        if not path.is_file():
            return False, f"Missing {name}"
        try:
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                return False, f"Invalid {name}"
        except (OSError, ValueError):
            return False, f"Invalid {name}"
    if not any((model_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        return False, "Tokenizer vocabulary is missing"
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_path / index_name
        if not index_path.is_file():
            continue
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                return False, "Empty or invalid weight map"
            shards = set(weight_map.values())
            for shard in shards:
                shard_path = Path(shard)
                if shard_path.is_absolute() or ".." in shard_path.parts:
                    return False, "Invalid shard path in model index"
                full_path = model_path / shard_path
                if not full_path.is_file() or full_path.stat().st_size == 0:
                    return False, f"Missing or empty model shard: {shard}"
        except (OSError, ValueError, KeyError, TypeError):
            return False, f"Invalid {index_name}"
        return True, f"Local model has {len(shards)} weight shards"
    if any(
        (model_path / name).is_file() and (model_path / name).stat().st_size > 0
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return True, "Local model has a single weight file"
    return False, "Model weights are absent or incomplete"


def check_model(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    local_path = config.get("model", {}).get("local_path") or os.environ.get("REPRO_MODEL_PATH")
    if not local_path:
        return _check(
            "model_weights", "deferred", "Model weights were intentionally not installed", required=False
        )
    if config.get("model", {}).get("backend", "transformers") == "llama_cpp":
        from repro.serving.gguf import inspect_gguf, validate_gguf_file

        path = _resolve(project_root, local_path)
        complete, detail = validate_gguf_file(path)
        metadata = inspect_gguf(path) if complete else None
        return _check(
            "model_weights",
            "ok" if complete else ("error" if path.exists() else "deferred"),
            detail,
            required=False,
            backend="llama_cpp",
            gguf=metadata,
            note="Readable local GGUF metadata is not proof of a successful inference run",
        )
    complete, detail = validate_model_directory(_resolve(project_root, local_path))
    return _check("model_weights", "ok" if complete else "deferred", detail, required=False)


def check_model_endpoint(config: Mapping[str, Any]) -> dict[str, Any]:
    import httpx

    model = config.get("model", {})
    base_url = str(model.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return _check(
            "endpoint",
            "error",
            "Endpoint must be an HTTP(S) URL without embedded credentials",
            required=False,
        )
    headers = {
        "Authorization": f"Bearer {os.environ.get(model.get('api_key_env', 'REPRO_API_KEY'), 'local-only')}"
    }
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            names = [item["id"] for item in response.json().get("data", [])]
            server_details: dict[str, Any] = {}
            if model.get("backend") == "llama_cpp":
                props_url = f"{parsed.scheme}://{parsed.netloc}/props"
                props_response = client.get(props_url, headers=headers)
                props_response.raise_for_status()
                props = props_response.json()
                defaults = props.get("default_generation_settings", {})
                params = defaults.get("params", {})
                kwargs = props.get("chat_template_kwargs", params.get("chat_template_kwargs", {}))
                if isinstance(kwargs, str):
                    try:
                        kwargs = json.loads(kwargs)
                    except ValueError:
                        kwargs = {}
                thinking = props.get("enable_thinking")
                if thinking is None and isinstance(kwargs, dict):
                    thinking = kwargs.get("enable_thinking")
                if not isinstance(thinking, bool):
                    thinking = None
                server_details = {
                    "backend": "llama_cpp",
                    "build_info": props.get("build_info"),
                    "reported_context_tokens": defaults.get("n_ctx"),
                    "total_slots": props.get("total_slots"),
                    "reported_enable_thinking": thinking,
                    "thinking_mode_verified": thinking is not None,
                    "reasoning_parser_format": params.get("reasoning_format"),
                    "reasoning_parser_in_content": params.get("reasoning_in_content"),
                }
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return _check(
            "endpoint", "deferred", f"Model service is unavailable ({type(exc).__name__})", required=False
        )
    expected = model.get("name")
    matches = not expected or expected in names
    if server_details and server_details["reported_context_tokens"] is not None:
        matches = matches and server_details["reported_context_tokens"] == model.get("max_context_tokens")
    if server_details and server_details["reported_enable_thinking"] is not None:
        matches = matches and server_details["reported_enable_thinking"] == model.get("enable_thinking")
    return _check(
        "endpoint",
        "ok" if matches else "error",
        "Configured model is served"
        if matches
        else "Service model, context, or thinking settings differ from configuration",
        required=False,
        models=names,
        **server_details,
    )


def collect_diagnostics(
    project_root: Path, config: Mapping[str, Any] | None = None, *, check_endpoint: bool = False
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config = config or {}
    checks = [
        check_python(),
        check_dependencies(),
        check_upstream(project_root),
        check_gpu(),
        check_wsl(),
        check_inference_runtime(project_root, config),
        check_data(project_root, config),
        check_model(project_root, config),
    ]
    if check_endpoint:
        checks.append(check_model_endpoint(config))
    ready = all(item["status"] == "ok" for item in checks if item["required"])
    model_ready = any(item["name"] == "model_weights" and item["status"] == "ok" for item in checks)
    return {
        "project_root": str(project_root),
        "platform": platform.platform(),
        "checks": checks,
        "infrastructure_ready": ready,
        "model_ready": model_ready,
        "experiment_ready": ready
        and model_ready
        and any(item["name"] == "endpoint" and item["status"] == "ok" for item in checks),
    }


def doctor_text(report: Mapping[str, Any]) -> str:
    lines = [f"Project: {report['project_root']}"]
    lines.extend(f"[{item['status']}] {item['name']}: {item['message']}" for item in report["checks"])
    lines.append(f"Infrastructure ready: {report['infrastructure_ready']}")
    lines.append(f"Model weights ready: {report['model_ready']}")
    lines.append(f"Experiment ready: {report['experiment_ready']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-weights", type=Path, required=True)
    args = parser.parse_args()
    complete, message = validate_model_directory(args.check_weights)
    print(message)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
