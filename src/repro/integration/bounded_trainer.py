"""Small subclass adaptations: bounded offspring scheduling and artifact exports."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from repro.integration.source import ensure_upstream

ensure_upstream()

import graphskillevo.trainer as upstream_trainer  # noqa: E402
from graphskillevo.trainer import EvolutionarySkillTrainer  # noqa: E402

from repro.artifacts import RunStore  # noqa: E402
from repro.integration.model_bridge import OptimizerBridge  # noqa: E402
from repro.visualization import lineage_to_mermaid, skill_to_mermaid  # noqa: E402


class BoundedTrainer(EvolutionarySkillTrainer):
    def __init__(
        self, *args, bridge: OptimizerBridge, store: RunStore,
        operator_schedule: list[str], new_candidates: int, max_operator_attempts: int = 6, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bridge, self.store = bridge, store
        self.schedule, self.new_candidates = operator_schedule, new_candidates
        self.max_operator_attempts = max_operator_attempts
        self._archived: set[str] = set()
        self._lineage: dict[str, dict[str, Any]] = {}

    def _initialize_population(self, operators):
        with self.bridge.phase("initialization", self.max_operator_attempts):
            return super()._initialize_population(operators)

    def _generate_offspring(self, operators, ranked_population, weights, *, generation):
        start = (generation - 1) * self.new_candidates
        schedule = [
            self.schedule[(start + index) % len(self.schedule)] for index in range(self.new_candidates)
        ]
        self.store.record_event("operator_schedule", generation=generation, operators=schedule)
        # Keep upstream generation, parent sampling, scope enforcement, and retry records intact.
        # Only replace the sequence for this invocation; retained population size remains independent.
        with (
            self.bridge.phase(f"generation_{generation}", self.max_operator_attempts * self.new_candidates),
            patch.object(upstream_trainer, "child_operator_sequence", lambda population_size: schedule),
        ):
            offspring, records = super()._generate_offspring(
                operators, ranked_population, weights, generation=generation,
            )
        for individual in offspring:
            self._archive(individual)
        return offspring, records

    def _archive(self, individual) -> None:
        record = individual.to_record()
        self._lineage[individual.id] = record
        if individual.id in self._archived:
            return
        self.store.save_skill(
            individual.id, individual.content,
            origin=individual.origin, generation=individual.generation, parent_ids=individual.parent_ids,
            upstream_hash=individual.hash,
        )
        self.store.save_text(f"graphs/{individual.id}.mmd", skill_to_mermaid(individual.content))
        self._archived.add(individual.id)

    def _save_population(self, population, *, generation):
        super()._save_population(population, generation=generation)
        for individual in population:
            self._archive(individual)
        if generation == 0:
            self.store.record_history({
                "generation": 0, "population": [item.to_record() for item in population],
            })
        elif self.history and self.history[-1]["generation"] == generation:
            record = self.history[-1]
            retained = {item.id for item in population}
            for child in record["offspring"]:
                self._lineage[child["id"]] = {**child, "retained": child["id"] in retained}
            self.store.record_history(record)

    def export_lineage(self) -> None:
        self.store.save_text("graphs/lineage.mmd", lineage_to_mermaid(self._lineage.values()))
