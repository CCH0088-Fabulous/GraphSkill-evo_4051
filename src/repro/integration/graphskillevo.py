"""Compose pinned upstream evolution with local data, models, and trace artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repro.artifacts import RunStore
from repro.config import PROJECT_ROOT, resolve_path, validate_config
from repro.datasets.io import load_split
from repro.integration.dataloader import QADataLoader
from repro.integration.model_bridge import OptimizerBridge, scoped_upstream
from repro.integration.qa_adapter import QAAdapter
from repro.integration.source import verify_upstream


def run_evolution(config: dict[str, Any], client: Any, store: RunStore) -> dict[str, Any]:
    """Run the original trainer with explicitly documented, shared course adaptations."""
    validate_config(config)
    condition = config["experiment"]["condition"]
    selector = config["evolution"]["selector"]
    if condition not in {"original", "improved", "ablation"}:
        raise ValueError("Evolution requires original, improved, or ablation condition")
    if condition == "original" and selector != "original":
        raise ValueError("The original condition must use the inspected original failure selector")
    if condition in {"improved", "ablation"} and selector == "original":
        raise ValueError("An improved or ablation condition requires B's explicitly registered selector")
    if client.ledger.summary()["limits"]["max_calls"] is None:
        raise ValueError("Evolution requires a finite model-call budget to bound upstream retry loops")
    source = verify_upstream()
    from graphskillevo.config import EvolutionConfig
    from graphskillevo.operators import EvolutionOperators

    from repro.integration.bounded_trainer import BoundedTrainer

    dataset = config["data"]["dataset"]
    split_dir = resolve_path(config["data"]["split_dir"])
    train, val = (load_split(split_dir, name, dataset) for name in ("train", "val"))
    if config["experiment"]["profile"] == "course" and (len(train) > 100 or len(val) > 30):
        raise ValueError("Course profile allows at most 100 training and 30 validation examples")
    evolution = config["evolution"]
    actual_train_n = min(evolution["train_eval_num"], len(train))
    if evolution["reflection_failure_k"] >= actual_train_n:
        raise ValueError("reflection_failure_k must be smaller than the actual training batch")
    initial = config["experiment"].get("initial_skill")
    if initial:
        initial_path = resolve_path(initial)
        initial_origin = "configured_initial"
    elif dataset == "searchqa":
        initial_path = PROJECT_ROOT / (
            "third_party/GraphSkillEvo/graphskillevo/initial_skills/searchqa/init_skill.md"
        )
        initial_origin = "upstream_searchqa_initial"
    else:
        initial_path = PROJECT_ROOT / "skills/hotpotqa/initial.md"
        initial_origin = "local_hotpotqa_initial"
    if not initial_path.is_file():
        raise FileNotFoundError(f"Initial skill is missing: {initial_path}")
    adaptations = {
        "source": source, "initial_skill_origin": initial_origin,
        "population_size": evolution["population_size"],
        "new_candidates_per_round": evolution["new_candidates_per_round"],
        "operator_schedule": evolution["operator_schedule"],
        "operator_schedule_policy": "rotate across generations, independent of retained population size",
        "max_operator_attempts_per_phase_unit": evolution.get("max_operator_attempts", 6),
        "failure_selector": evolution["selector"],
        "optimizer_seed_policy": config["model"]["seed_policy"],
        "truncation_policy": "QA: explicit zero-score failure retained; optimizer: reject and bounded retry",
        "model_routing": "one metered client, optimizer and target roles; no upstream Azure configuration",
        "searchqa_prompt": "upstream prompt and 6000-character context truncation",
        "hotpotqa_prompt": "same QA system prompt, all supplied context passages retained",
        "test_access_during_evolution": False,
        "training_feedback": (
            "upstream failure reason and scored trajectory format; no added hidden reference"
        ),
        "validation_cache": "upstream within-run skill-content cache; reuse is not a fresh independent run",
    }
    store.save_json("integration_adaptations.json", adaptations)
    cfg = {
        "out_root": str(store.path / "upstream"), "env": dataset,
        "seed": config["experiment"]["seed"], "skill_init": str(initial_path),
        "target_model": config["model"]["name"], "optimizer_model": config["model"]["name"],
    }
    evolution_cfg = EvolutionConfig(
        population_size=evolution["population_size"], generations=evolution["generations"],
        train_eval_num=evolution["train_eval_num"], reflection_failure_k=evolution["reflection_failure_k"],
        include_hidden_reference=True,
        operator_max_completion_tokens=evolution["operator_max_completion_tokens"],
        operator_reasoning_effort="", initial_include_base_skill=True,
        graph_retry_attempts=evolution["graph_retry_attempts"],
    )
    loader = QADataLoader(train, val, dataset=dataset)
    adapter = QAAdapter(
        loader, client, store, config["experiment"]["condition"], config["model"]["max_completion_tokens"],
    )
    operators = EvolutionOperators(
        env_name=dataset, max_completion_tokens=evolution["operator_max_completion_tokens"],
        reasoning_effort=None, graph_retry_attempts=evolution["graph_retry_attempts"],
    )
    bridge = OptimizerBridge(client, store, seed_policy=config["model"]["seed_policy"])
    trainer = BoundedTrainer(
        cfg, evolution_cfg, adapter, operators=operators, bridge=bridge, store=store,
        operator_schedule=evolution["operator_schedule"],
        new_candidates=evolution["new_candidates_per_round"],
        max_operator_attempts=evolution.get("max_operator_attempts", 6),
    )
    with scoped_upstream(bridge, selector_name=evolution["selector"], dataset=dataset):
        summary = trainer.train()
    trainer.export_lineage()
    return {
        **summary, "best_skill_path": str(Path(cfg["out_root"]) / "best_skill.md"),
        "adaptations": adaptations, "accounting": client.ledger.summary(),
    }
