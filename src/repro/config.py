"""Load experiment settings and reject invalid or ambiguous runs before requests."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ("mutation_other", "mutation_graph", "crossover_other", "crossover_graph")
CONDITIONS = ("no_skill", "initial_skill", "original", "improved", "ablation")

DEFAULTS: dict[str, Any] = {
    "model": {
        "backend": "transformers",
        "name": "Qwen/Qwen3.5-9B",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "REPRO_API_KEY",
        "local_path": "models/Qwen3.5-9B",
        "max_completion_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 50,
        "seed_policy": "fixed",
        "timeout_s": 180,
        "max_retries": 1,
        "enable_thinking": False,
        "max_context_tokens": 8192,
    },
    "data": {"dataset": "searchqa", "split_dir": "data/processed/searchqa", "manifest": None},
    "experiment": {
        "condition": "original",
        "seed": 42,
        "output_root": "outputs",
        "allow_test": False,
        "profile": "course",
        "initial_skill": None,
    },
    "evolution": {
        "generations": 1,
        "population_size": 2,
        "new_candidates_per_round": 2,
        "train_eval_num": 5,
        "reflection_failure_k": 2,
        "selector": "original",
        "operator_schedule": ["mutation_other", "mutation_graph", "crossover_other", "crossover_graph"],
        "operator_max_completion_tokens": 4096,
        "graph_retry_attempts": 2,
        "max_operator_attempts": 6,
    },
    "budget": {"max_calls": 200, "max_completion_tokens": 200000, "gpu_hours_reference": 40},
    "preparation": {
        "raw_cache_dir": "data/raw",
        "searchqa_manifest_dir": "third_party/GraphSkillEvo/data/searchqa_id_split",
        "searchqa_revision": None,
        "hotpotqa_revision": None,
        "searchqa_download_mode": "parquet",
        "hotpotqa_download_mode": "parquet",
        "hotpotqa_source": "official-public",
        "searchqa_raw_path": None,
        "hotpotqa_raw_train": None,
        "hotpotqa_raw_val": None,
        "hotpotqa_raw_test": None,
        "seed": 42,
        "train_count": 100,
        "val_count": 30,
        "test_count": 50,
        "pilot_seed": 42,
        "pilot_train_count": 10,
        "pilot_val_count": 5,
    },
}


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _merge_with_data_reference(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = _merge(base, extra)
    reference = base.get("data", {}).get("manifest")
    if reference and result.get("data", {}).get("manifest") != reference:
        raise ValueError("A declared data manifest cannot be overridden or disabled")
    return result


def _validate_layer(raw: Any, path: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a nonempty YAML mapping")
    if not raw:
        raise ValueError(f"{path}: config must not be empty")
    unknown = set(raw) - set(DEFAULTS) - {"_base_", "_presets_", "presets"}
    if unknown:
        raise ValueError(f"{path}: unknown config sections: {sorted(map(str, unknown))}")
    for section, values in raw.items():
        if section in {"_base_", "_presets_"}:
            continue
        if section == "presets":
            if not isinstance(values, dict):
                raise ValueError(f"{path}: presets must be a mapping")
            for name, preset in values.items():
                if not isinstance(name, str) or not name.strip() or not isinstance(preset, dict):
                    raise ValueError(f"{path}: preset names must be nonempty strings with mapping values")
                if set(preset) - set(DEFAULTS):
                    raise ValueError(f"{path}: preset {name} has unknown sections")
                if preset:
                    _validate_layer(preset, path)
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{path}: {section} must be a mapping")
        if set(values) - set(DEFAULTS[section]):
            unknown_options = sorted(map(str, set(values) - set(DEFAULTS[section])))
            raise ValueError(f"{path}: unknown {section} options: {unknown_options}")
    return raw


def _read_layer(path: Path, ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"Config inheritance cycle: {chain}")
    try:
        raw = _validate_layer(yaml.safe_load(path.read_text(encoding="utf-8-sig")), path)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    result: dict[str, Any] = {}
    if "_base_" in raw:
        base = raw["_base_"]
        if not isinstance(base, str) or not base.strip():
            raise ValueError(f"{path}: _base_ must be a nonempty file path string")
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        result = _read_layer(base_path, (*ancestors, path))
    if "presets" in raw:
        result["presets"] = _merge(result.get("presets", {}), raw["presets"])
    result = _apply_presets(result, raw.get("_presets_", []), path)
    layer = {key: value for key, value in raw.items() if key not in {"_base_", "_presets_", "presets"}}
    return _merge_with_data_reference(result, layer)


def _apply_presets(config: dict[str, Any], names: Any, path: Path) -> dict[str, Any]:
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{path}: _presets_ must be a list of nonempty names")
    result = deepcopy(config)
    for name in names:
        preset = result.get("presets", {}).get(name)
        if preset is None:
            raise ValueError(f"{path}: unknown preset {name!r}")
        result = _merge_with_data_reference(result, preset)
    return result


def load_config(
    path: str | Path, overrides: list[str] | None = None, *, presets: list[str] | None = None
) -> dict[str, Any]:
    """Merge explicit YAML inheritance, environment overrides, then CLI overrides."""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    layer = _apply_presets(_read_layer(config_path), presets or [], config_path)
    layer.pop("presets", None)
    config = _merge(DEFAULTS, layer)
    declared_data_manifest = config["data"].get("manifest")
    environment = {**dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
    for env_name, option in (
        ("REPRO_MODEL_NAME", "name"),
        ("REPRO_BASE_URL", "base_url"),
        ("REPRO_MODEL_PATH", "local_path"),
    ):
        if environment.get(env_name):
            config["model"][option] = environment[env_name]
    for override in overrides or []:
        name, sep, text = override.partition("=")
        parts = name.split(".")
        if not sep or len(parts) != 2 or parts[0] not in config or parts[1] not in config[parts[0]]:
            raise ValueError(f"Unknown override {override!r}; expected section.option=value")
        config[parts[0]][parts[1]] = yaml.safe_load(text)
    if declared_data_manifest and config["data"].get("manifest") != declared_data_manifest:
        raise ValueError("A declared data manifest cannot be overridden or disabled")
    validate_config(config)
    return config


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _positive_int(value: Any, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")


def validate_config(config: dict[str, Any]) -> None:
    experiment, evolution, model = config["experiment"], config["evolution"], config["model"]
    preparation = config["preparation"]
    for key in (
        "seed", "train_count", "val_count", "test_count", "pilot_seed", "pilot_train_count", "pilot_val_count"
    ):
        _positive_int(preparation[key], f"preparation.{key}", zero=True)
    for key in ("raw_cache_dir", "searchqa_manifest_dir"):
        if not isinstance(preparation[key], str) or not preparation[key].strip():
            raise ValueError(f"preparation.{key} must be a nonempty path")
    for key in ("searchqa_revision", "hotpotqa_revision"):
        value = preparation[key]
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value)):
            raise ValueError(f"preparation.{key} must be a fixed 40-character commit revision")
    for key in ("searchqa_raw_path", "hotpotqa_raw_train", "hotpotqa_raw_val", "hotpotqa_raw_test"):
        value = preparation[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"preparation.{key} must be a nonempty path or null")
    if preparation["searchqa_download_mode"] not in {"parquet", "viewer", "streaming"}:
        raise ValueError("Unsupported preparation.searchqa_download_mode")
    if preparation["hotpotqa_download_mode"] not in {"parquet", "viewer"}:
        raise ValueError("Unsupported preparation.hotpotqa_download_mode")
    if preparation["hotpotqa_source"] not in {"official-public", "course"}:
        raise ValueError("Unsupported preparation.hotpotqa_source")
    if config["data"]["dataset"] not in {"searchqa", "hotpotqa"}:
        raise ValueError("data.dataset must be searchqa or hotpotqa")
    if experiment["condition"] not in CONDITIONS:
        raise ValueError(f"Unknown condition: {experiment['condition']}")
    if experiment["profile"] not in {"course", "custom", "synthetic"}:
        raise ValueError("experiment.profile must be course, custom, or synthetic")
    if not isinstance(experiment["allow_test"], bool):
        raise ValueError("experiment.allow_test must be boolean")
    for key in (
        "population_size",
        "new_candidates_per_round",
        "train_eval_num",
        "reflection_failure_k",
        "operator_max_completion_tokens",
        "graph_retry_attempts",
        "max_operator_attempts",
    ):
        _positive_int(evolution[key], f"evolution.{key}", zero=key == "reflection_failure_k")
    _positive_int(evolution["generations"], "evolution.generations", zero=True)
    _positive_int(experiment["seed"], "experiment.seed", zero=True)
    if model["backend"] not in {"transformers", "llama_cpp"}:
        raise ValueError("model.backend must be transformers or llama_cpp")
    for key in ("max_completion_tokens", "max_context_tokens", "timeout_s", "max_retries"):
        _positive_int(model[key], f"model.{key}", zero=key == "max_retries")
    if model["max_completion_tokens"] >= model["max_context_tokens"]:
        raise ValueError("model.max_completion_tokens must leave room for input within max_context_tokens")
    if evolution["operator_max_completion_tokens"] >= model["max_context_tokens"]:
        raise ValueError("evolution.operator_max_completion_tokens must leave room for input in the context")
    if not isinstance(model["enable_thinking"], bool):
        raise ValueError("model.enable_thinking must be boolean")
    if not isinstance(model["temperature"], (int, float)) or not 0 <= model["temperature"] <= 2:
        raise ValueError("model.temperature must be between 0 and 2")
    if not isinstance(model["top_p"], (int, float)) or not 0 < model["top_p"] <= 1:
        raise ValueError("model.top_p must be in (0, 1]")
    _positive_int(model["top_k"], "model.top_k", zero=True)
    if model["seed_policy"] not in {"fixed", "optimizer_attempt_v1", "optimizer_retry_v1"}:
        raise ValueError("Unsupported model.seed_policy")
    if evolution["reflection_failure_k"] >= evolution["train_eval_num"]:
        raise ValueError("reflection_failure_k must be smaller than train_eval_num")
    schedule = evolution["operator_schedule"]
    if not isinstance(schedule, list) or not schedule or any(item not in OPERATORS for item in schedule):
        raise ValueError(f"operator_schedule must be a nonempty list drawn from {OPERATORS}")
    if experiment["profile"] == "course":
        if evolution["generations"] > 3 or evolution["new_candidates_per_round"] > 2:
            raise ValueError("Course profile allows at most 3 rounds and 2 new candidates per round")
    condition = experiment["condition"]
    if condition == "original" and evolution["selector"] != "original":
        raise ValueError("The original condition must use evolution.selector=original")
    if condition in {"improved", "ablation"} and evolution["selector"] == "original":
        raise ValueError(
            f"The {condition} condition requires B's registered selector; original is not a fallback"
        )
    for key in ("max_calls", "max_completion_tokens"):
        if config["budget"][key] is not None:
            _positive_int(config["budget"][key], f"budget.{key}")
    if (
        experiment["condition"] in {"original", "improved", "ablation"}
        and config["budget"]["max_calls"] is None
    ):
        raise ValueError("Evolution requires a finite max_calls budget to bound upstream generation retries")
    if not model["name"] or not model["base_url"].startswith(("http://", "https://")):
        raise ValueError("Model name and HTTP(S) base_url are required")
