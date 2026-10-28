"""Общие заготовки для тестов М1: мир, задачи миссии и снимок группы без симулятора."""

from __future__ import annotations

import numpy as np

from swarmsim.interfaces import AgentView, SwarmSnapshot
from swarmsim.missions import decompose
from swarmsim.scenario import Scenario
from swarmsim.world import World


def make_world_tasks(size=(1000, 1000), n_cells=16, nofly=(), swath=40.0, area=None):
    sc = Scenario.model_validate({
        "world": {"size": list(size), "nofly": [list(map(list, z)) for z in nofly], "targets": {"count": 0}},
        "sensor": {"swath_width": swath},
        "mission": {"n_cells": n_cells, **({"area": area} if area else {})},
    })
    world = World.from_config(sc.world, np.random.default_rng(0))
    if sc.mission.swath_width is None:
        sc.mission.swath_width = swath
    return world, decompose(sc.mission, world)


def snapshot(world, positions, *, alive=None, homes=None, current=None, queues=None, speed=10.0, used=None):
    alive = alive or {}
    agents = {
        i: AgentView(
            agent_id=i, pos=tuple(p), home=tuple((homes or {}).get(i, p)), alive=alive.get(i, True), speed=speed,
            current_task=(current or {}).get(i), queue=list((queues or {}).get(i, [])),
            energy_used_wh=(used or {}).get(i, 0.0),
        )
        for i, p in positions.items()
    }
    return SwarmSnapshot(t=0.0, agents=agents, world=world)


def owners(assignment):
    out: dict[str, list[int]] = {}
    for aid, plan in assignment.plans.items():
        for tid in plan:
            out.setdefault(tid, []).append(aid)
    return out
