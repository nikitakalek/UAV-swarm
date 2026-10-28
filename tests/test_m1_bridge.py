"""Распределители М1 работают внутри симулятора М2 через адаптер ``swarmsim.m1_bridge``."""

import pytest

from swarmsim.sim import run


@pytest.mark.parametrize("name", ["static_naive", "static_voronoi", "central_greedy"])
def test_nominal_all_tasks_done(scenario_factory, name):
    sc = scenario_factory(allocator={"name": name}, mission={"n_cells": 16})
    m = run(sc, seed=1).metrics
    assert m["CR"] == 1.0
    assert m["N_dup"] == 0


def test_central_greedy_recovers_after_failures(scenario_factory):
    sc = scenario_factory(
        allocator={"name": "central_greedy"},
        agents={"count": 6},
        mission={"n_cells": 16},
        faults={"detect_delay": 2.0},
        events=[{"t": 120, "type": "agent_failure", "agent": 1},
                {"t": 200, "type": "agent_failure", "agent": 4}],
    )
    m = run(sc, seed=1).metrics
    assert m["CR"] == 1.0
    assert m["N_dup"] == 0
    assert m["N_lost"] == 0


def test_static_does_not_repair(scenario_factory):
    """B1/B1b намеренно без перераспределения: задачи отказавшего остаются невыполненными."""
    sc = scenario_factory(
        allocator={"name": "static_voronoi"},
        mission={"n_cells": 16},
        events=[{"t": 60, "type": "agent_failure", "agent": 0}],
    )
    assert run(sc, seed=1).metrics["CR"] < 1.0


def test_deterministic(scenario_factory):
    sc = scenario_factory(allocator={"name": "central_greedy"},
                          events=[{"t": 100, "type": "agent_failure", "agent": 2}])
    def model(m):  # без замеров настенного времени
        return {k: v for k, v in m.items() if not k.startswith("wall_")}

    assert model(run(sc, seed=3).metrics) == model(run(sc, seed=3).metrics)
