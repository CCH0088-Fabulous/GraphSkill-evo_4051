"""Optional NVIDIA sampling; importing this module never starts a process."""

from __future__ import annotations

import csv
import io
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from repro.artifacts.serialization import utc_now


def sample_gpu() -> list[dict[str, Any]]:
    """Read utilization, memory, and power without requiring PyTorch or weights."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, timeout=5,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    rows = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("Unexpected nvidia-smi output")
        index, name, total, used, utilization, power = [value.strip() for value in row]
        rows.append({
            "timestamp": utc_now(), "index": int(index), "name": name,
            "memory_total_mib": _number(total), "memory_used_mib": _number(used),
            "utilization_percent": _number(utilization), "power_w": _number(power),
        })
    return rows


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


class GPUMonitor:
    """A context-managed sampling thread; monitor time is not GPU compute time."""

    def __init__(self, sink: Callable[[dict[str, Any]], None], interval_s: float = 5) -> None:
        if interval_s <= 0:
            raise ValueError("GPU sampling interval must be positive")
        self.sink = sink
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def __enter__(self) -> GPUMonitor:
        if self._thread is not None:
            raise RuntimeError("GPU monitor can only be started once")
        self._start = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for row in sample_gpu():
                    self.sink({"event": "gpu_sample", **row})
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.sink({"event": "gpu_sample_error", "timestamp": utc_now(), "error": str(exc)})
            self._stop.wait(self.interval_s)

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        self.sink({
            "event": "gpu_monitor_stopped", "timestamp": utc_now(),
            "monitor_wall_time_s": time.monotonic() - self._start,
            "note": "Sampling duration; not measured CUDA kernel time or proof of exclusive GPU allocation.",
        })
