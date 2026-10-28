"""SectorAllocator и порядок обхода (М1): полнота, баланс, компактность, ремонт после отказа."""

from __future__ import annotations

import math

import pytest

from m1_helpers import make_world_tasks, owners, snapshot
from swarmsim.alloc import SectorAllocator
from swarmsim.missions.base import Task
from swarmsim.planning.routing import order_tasks, route_cost
from swarmsim.registry import make_allocator
from swarmsim.sim import run

CORNERS = {0: (0, 0), 1: (1000, 0), 2: (0, 1000), 3: (1000, 1000)}


def _partition_ok(asg, tasks, alive):
    own = owners(asg)
    assert set(own) == {t.id for t in tasks.unfinished()}, "все незавершённые задачи назначены"
    assert all(len(v) == 1 for v in own.values()), "нет дублей"
    assert set(asg.plans) == set(alive)


# ------------------------------------------------------------------ начальное распределение
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_initial_assigns_every_task_once(n):
    world, tasks = make_world_tasks(n_cells=25)
    pos = {i: (i * 5.0, 0.0) for i in range(n)}
    asg = SectorAllocator().allocate(snapshot(world, pos), tasks)
    _partition_ok(asg, tasks, list(pos))
    assert asg.meta["mode"] == "initial"


def test_sectors_go_to_nearest_agents():
    """Аппарат в углу получает задачи своего угла: секторы сопоставлены по близости."""
    world, tasks = make_world_tasks(n_cells=16)
    asg = SectorAllocator().allocate(snapshot(world, CORNERS), tasks)
    for aid, plan in asg.plans.items():
        cx = sum(tasks[t].centroid[0] for t in plan) / len(plan)
        cy = sum(tasks[t].centroid[1] for t in plan) / len(plan)
        assert math.dist((cx, cy), CORNERS[aid]) < 500


def test_load_is_balanced():
    world, tasks = make_world_tasks(n_cells=36)
    asg = SectorAllocator().allocate(snapshot(world, CORNERS), tasks)
    loads = asg.meta["load_m"]
    assert max(loads.values()) / min(loads.values()) < 1.35


def test_balance_reduces_peak_load():
    world, tasks = make_world_tasks(n_cells=36)
    pos = {i: (i * 5.0, 0.0) for i in range(5)}  # все с одной площадки — дальний сектор перегружен
    raw = SectorAllocator(balance=False).allocate(snapshot(world, pos), tasks).meta["load_m"]
    bal = SectorAllocator().allocate(snapshot(world, pos), tasks).meta["load_m"]
    assert max(bal.values()) <= max(raw.values())


def test_deterministic():
    world, tasks = make_world_tasks(n_cells=25)
    a = SectorAllocator().allocate(snapshot(world, CORNERS), tasks)
    b = SectorAllocator().allocate(snapshot(world, CORNERS), tasks)
    assert a.plans == b.plans


def test_more_agents_than_tasks():
    world, tasks = make_world_tasks(n_cells=4)
    pos = {i: (i * 100.0, 0.0) for i in range(6)}
    asg = SectorAllocator().allocate(snapshot(world, pos), tasks)
    _partition_ok(asg, tasks, list(pos))


def test_nofly_domain():
    world, tasks = make_world_tasks(n_cells=25, nofly=[[(400, 400), (600, 400), (600, 600), (400, 600)]])
    asg = SectorAllocator().allocate(snapshot(world, CORNERS), tasks)
    _partition_ok(asg, tasks, list(CORNERS))


# ------------------------------------------------------------------ ремонт
def _initial(world, tasks, pos):
    alloc = SectorAllocator()
    asg = alloc.allocate(snapshot(world, pos), tasks)
    return alloc, asg


def test_repair_keeps_survivors_plans_and_splits_orphans():
    world, tasks = make_world_tasks(n_cells=36)
    alloc, asg = _initial(world, tasks, CORNERS)
    plans = {i: list(p) for i, p in asg.plans.items()}
    snap = snapshot(world, CORNERS, alive={2: False},
                    current={i: plans[i][0] for i in plans}, queues={i: plans[i][1:] for i in plans})
    rep = alloc.allocate(snap, tasks)
    _partition_ok(rep, tasks, [0, 1, 3])
    assert rep.meta["mode"] == "repair"
    for i in (0, 1, 3):
        assert rep.plans[i][0] == plans[i][0], "текущая задача не отбирается"
    orphans = set(plans[2])
    takers = {i for i in (0, 1, 3) if orphans & set(rep.plans[i])}
    assert len(takers) >= 2, "сироты делятся между несколькими аппаратами"
    assert all("свободная задача" in rep.reasons[t] or "выравнивание" in rep.reasons[t] for t in orphans)


def test_repair_prefers_nearest_agent():
    """Сироты из угла выбывшего достаются в основном его соседям, а не дальнему аппарату."""
    world, tasks = make_world_tasks(n_cells=36)
    alloc, asg = _initial(world, tasks, CORNERS)
    plans = {i: list(p) for i, p in asg.plans.items()}
    snap = snapshot(world, CORNERS, alive={0: False}, queues=plans)
    rep = alloc.allocate(snap, tasks)
    got = {i: len(set(plans[0]) & set(rep.plans[i])) for i in (1, 2, 3)}
    assert got[3] <= min(got[1], got[2])


def test_no_repair_mode_leaves_orphans():
    world, tasks = make_world_tasks(n_cells=16)
    alloc = SectorAllocator(repair=False)
    asg = alloc.allocate(snapshot(world, CORNERS), tasks)
    snap = snapshot(world, CORNERS, alive={1: False}, queues=asg.plans)
    rep = alloc.allocate(snap, tasks)
    assert not set(asg.plans[1]) & set(owners(rep))


def test_new_task_is_assigned():
    world, tasks = make_world_tasks(n_cells=16)
    alloc, asg = _initial(world, tasks, CORNERS)
    tasks.add(Task(id="extra_00", kind="resweep", waypoints=[(900, 900), (950, 950)]))
    rep = alloc.allocate(snapshot(world, CORNERS, queues=asg.plans), tasks)
    _partition_ok(rep, tasks, list(CORNERS))
    assert rep.meta["mode"] == "repair" and "свободная задача" in rep.reasons["extra_00"]


# ------------------------------------------------------------------ порядок обхода
def test_order_keeps_current_first_and_is_not_worse_than_nn():
    world, tasks = make_world_tasks(n_cells=25)
    ts = list(tasks)[:10]
    est = order_tasks(ts, (0, 0), (0, 0), first=ts[5].id)
    assert est.order[0] == ts[5].id
    assert sorted(est.order) == sorted(t.id for t in ts)
    nn = order_tasks(ts, (0, 0), (0, 0), first=ts[5].id, max_passes=0)
    assert est.length <= nn.length + 1e-9
    by_id = {t.id: t for t in ts}
    assert sum(route_cost([by_id[t] for t in est.order], (0, 0), (0, 0))) == pytest.approx(est.length)


# ------------------------------------------------------------------ в симуляторе
def test_registered_in_simulator():
    assert isinstance(make_allocator("sector"), SectorAllocator)


@pytest.mark.parametrize("events", [[], [{"t": 100, "type": "agent_failure", "agent": 1},
                                         {"t": 160, "type": "agent_failure", "agent": 3}]])
def test_simulation_completes_without_duplicates(scenario_factory, events):
    sc = scenario_factory(allocator={"name": "sector"}, agents={"count": 5}, mission={"n_cells": 25},
                          events=events)
    m = run(sc, seed=1).metrics
    assert m["CR"] == 1.0 and m["N_dup"] == 0 and m["N_lost"] == 0 and m["N_col"] == 0


def test_static_mode_loses_failed_agent_work(scenario_factory):
    sc = scenario_factory(allocator={"name": "sector", "params": {"repair": False}}, mission={"n_cells": 16},
                          events=[{"t": 60, "type": "agent_failure", "agent": 0}])
    assert run(sc, seed=1).metrics["CR"] < 1.0
