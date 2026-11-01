"""Scoped adaptations around the pinned, unchanged upstream implementation."""


def run_evolution(config, client, store):
    from repro.integration.graphskillevo import run_evolution as run

    return run(config, client, store)


__all__ = ["run_evolution"]
