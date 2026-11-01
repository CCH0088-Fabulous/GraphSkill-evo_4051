"""Install a checksum-pinned official Windows CUDA runtime; never download model weights."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "b10901"
COMMIT = "28ff0958291ce3465fabd7bd679d4b0edd742bd9"
ASSETS = {
    "llama-b10901-bin-win-cuda-13.3-x64.zip": (
        "dd3035688b6f3eaaabd41dddd5d66389fed3b6207107170dbefd884b80ad0d79"
    ),
    "cudart-llama-bin-win-cuda-13.3-x64.zip": (
        "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(filename: str, expected: str) -> Path:
    cache = ROOT / ".cache/llama.cpp"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / filename
    if path.is_file() and sha256(path) == expected:
        return path
    temporary = path.with_suffix(".zip.partial")
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{filename}"
    print(f"Downloading {filename}", flush=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    stream.write(chunk)
    if sha256(temporary) != expected:
        raise ValueError(f"Official SHA-256 mismatch: {filename}")
    temporary.replace(path)
    return path


def extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or ":" in member.filename:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Runtime archives cannot contain symbolic links")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with bundle.open(member) as stream:
                    archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if sha256(target) != archive_hash:
                    raise FileExistsError(f"Existing runtime file differs: {target}")
                continue
            with bundle.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def main() -> None:
    runtime = ROOT / "tools/llama.cpp" / f"{RELEASE}-cuda13.3"
    for name, digest in ASSETS.items():
        extract(download(name, digest), runtime)
    servers = list(runtime.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise ValueError("Expected exactly one llama-server.exe in the pinned runtime")
    lock = {
        "repository": "https://github.com/ggml-org/llama.cpp",
        "release": RELEASE,
        "commit": COMMIT,
        "cuda": "13.3",
        "platform": "win-x64",
        "executable": servers[0].relative_to(ROOT).as_posix(),
        "executable_sha256": sha256(servers[0]),
        "archives": ASSETS,
    }
    (runtime.parent / "runtime.lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
