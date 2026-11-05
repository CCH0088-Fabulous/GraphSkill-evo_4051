"""An append-only result store, separate from experiment and scoring logic."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import secrets
import threading
from pathlib import Path
from typing import Any

from repro.artifacts.serialization import json_text, utc_now
from repro.contracts import Prediction

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RunStore:
    """Create one run directory and never replace an existing run or artifact."""

    def __init__(self, root: str | Path, run_id: str | None = None) -> None:
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or (
            dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + secrets.token_hex(3)
        )
        if not _IDENTIFIER.fullmatch(self.run_id):
            raise ValueError("run_id must be a plain identifier, not a filesystem path")
        self.path = root / self.run_id
        self.path.mkdir(exist_ok=False)
        self._lock = threading.RLock()
        self._predictions: set[tuple[str, ...]] = set()

    @classmethod
    def open_existing(cls, path: str | Path) -> RunStore:
        """Reopen a directory for further new artifacts; existing files are still never replaced."""
        target = Path(path).resolve()
        if not target.is_dir() or not _IDENTIFIER.fullmatch(target.name):
            raise ValueError("Existing run directory required")
        store = cls.__new__(cls)
        store.run_id = target.name
        store.path = target
        store._lock = threading.RLock()
        store._predictions = set()
        return store

    def _target(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or candidate.drive:
            raise ValueError("Artifact paths must be relative to the run directory")
        if any(":" in part for part in candidate.parts):
            raise ValueError("Artifact paths cannot contain alternate data stream names")
        target = (self.path / candidate).resolve()
        if not target.is_relative_to(self.path.resolve()) or target == self.path.resolve():
            raise ValueError("Artifact path escapes the run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_json(self, relative: str | Path, value: Any) -> Path:
        payload = json_text(value, pretty=True) + "\n"
        with self._lock:
            target = self._target(relative)
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
        return target

    def save_text(self, relative: str | Path, content: str) -> Path:
        with self._lock:
            target = self._target(relative)
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        return target

    def append_jsonl(self, relative: str | Path, value: Any) -> Path:
        payload = json_text(value) + "\n"
        with self._lock:
            target = self._target(relative)
            with target.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
        return target

    def save_snapshot(self, config: Any, environment: Any, source: Any) -> None:
        self.save_json("config.json", config)
        self.save_json("environment.json", environment)
        self.save_json("source.json", source)
        self.record_event("run_created", run_id=self.run_id)

    def record_event(self, event: str, **fields: Any) -> None:
        self.append_jsonl("events.jsonl", {**fields, "event": event, "timestamp": utc_now()})

    def write_prediction(self, prediction: Prediction | dict[str, Any]) -> None:
        row = prediction.to_dict() if isinstance(prediction, Prediction) else dict(prediction)
        key = tuple(str(row[name]) for name in ("dataset", "split", "condition", "skill_id", "sample_id"))
        key += (str(row.get("evaluation_id", "")),)
        with self._lock:
            if key in self._predictions:
                raise ValueError(f"Duplicate prediction in run: {key}")
            self.append_jsonl("predictions.jsonl", row)
            self._predictions.add(key)

    def save_skill(self, skill_id: str, content: str, **metadata: Any) -> dict[str, Any]:
        if not _IDENTIFIER.fullmatch(skill_id):
            raise ValueError("skill_id must be a plain identifier")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        relative = Path("skills") / f"{skill_id}.md"
        record = {**metadata, "skill_id": skill_id, "sha256": digest, "path": relative.as_posix()}
        with self._lock:
            target = self._target(relative)
            metadata_target = self._target(relative.with_suffix(".json"))
            if target.exists() or metadata_target.exists():
                raise FileExistsError(f"Skill artifact already exists: {skill_id}")
            self.save_text(relative, content)
            self.save_json(relative.with_suffix(".json"), record)
            self.append_jsonl("skills.jsonl", record)
        return record

    def record_history(self, record: dict[str, Any]) -> None:
        """Append parent IDs, generation, validation scores, and retention decisions."""
        self.append_jsonl("history.jsonl", {**record, "timestamp": utc_now()})
