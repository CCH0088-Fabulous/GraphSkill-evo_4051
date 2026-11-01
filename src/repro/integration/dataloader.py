"""Reproduce upstream batch planning over prevalidated, normalized partitions."""

from __future__ import annotations

import random
from typing import Any

from repro.contracts import QAExample
from repro.datasets.manifests import validate_partitions
from repro.integration.source import ensure_upstream


class QADataLoader:
    def __init__(self, train: list[QAExample], val: list[QAExample], *, dataset: str) -> None:
        validate_partitions({"train": train, "val": val}, dataset=dataset)
        if not train or not val:
            raise ValueError("Evolution requires nonempty training and validation partitions")
        self.train_items = [example.to_dict() for example in train]
        self.val_items = [example.to_dict() for example in val]

    def setup(self, cfg: dict[str, Any]) -> None:
        pass

    def get_train_size(self) -> int:
        return len(self.train_items)

    def build_train_batch(self, batch_size: int, seed: int, **kwargs):
        ensure_upstream()
        from graphskillevo.datasets.base import BatchSpec

        items = list(self.train_items)
        random.Random(seed).shuffle(items)
        items = items[:batch_size]
        return BatchSpec(phase="train", split="train", seed=seed, batch_size=len(items), payload=items)

    def build_eval_batch(self, env_num: int, split: str, seed: int, **kwargs):
        ensure_upstream()
        from graphskillevo.datasets.base import BatchSpec

        if split not in {"val", "valid_seen", "selection"}:
            raise ValueError("Evolution dataloader exposes validation only; test is held out")
        items = list(self.val_items)
        if env_num and env_num < len(items):
            items = items[:env_num]
        return BatchSpec(phase="eval", split=split, seed=seed, batch_size=len(items), payload=items)
