"""PlanValidator (М1): каждое правило ловит своё нарушение, допустимый план проходит."""

from __future__ import annotations

from m1_helpers import make_world_tasks, snapshot
from swarmsim.alloc import SectorAllocator
from swarmsim.interfaces import Assignment
from swarmsim.planning.validator import PlanValidator, planned_path

CORNERS = {0: (0, 0), 1: (1000, 0), 2: (0, 1000), 3: (1000, 1000)}
LAYERS = {0: 40.0, 1: 45.0, 2: 50.0, 3: 55.0}
ZONE = [(400, 400), (600, 400), (600, 600), (400, 600)]


def _plan(world, tasks, pos=CORNERS):
    snap = snapshot(world, pos)
    return snap, SectorAllocator().allocate(snap, tasks)


def test_allocator_plan_is_valid():
    world, tasks = make_world_tasks(n_cells=25)
    snap, asg = _plan(world, tasks)
    rep = PlanValidator(battery_wh=400).validate(asg, snap, tasks, LAYERS)
    assert rep.ok, rep.describe()


def test_duplicate_and_missing_tasks():
    world, tasks = make_world_tasks(n_cells=4)
    ids = [t.id for t in tasks]
    asg = Assignment(plans={0: ids[:2], 1: [ids[1]]})
    rep = PlanValidator().validate(asg, snapshot(world, {0: (0, 0), 1: (10, 0)}), tasks)
    msgs = [str(v) for v in rep.violations if v.rule == "assignment"]
    assert any("дважды" in m for m in msgs)
    assert any("без исполнителя" in m for m in msgs)


def test_task_on_failed_agent():
    world, tasks = make_world_tasks(n_cells=4)
    asg = Assignment(plans={0: [t.id for t in tasks]})
    rep = PlanValidator().validate(asg, snapshot(world, {0: (0, 0)}, alive={0: False}), tasks)
    assert rep.by_rule().get("assignment")


def test_energy_shortfall():
    world, tasks = make_world_tasks(n_cells=9)
    snap, asg = _plan(world, tasks, {0: (0, 0)})
    rep = PlanValidator(battery_wh=20).validate(asg, snap, tasks)
    assert rep.by_rule() == {"energy": 1}
    # уже потраченная энергия уменьшает остаток
    ok = PlanValidator(battery_wh=400).validate(asg, snap, tasks)
    spent = PlanValidator(battery_wh=400).validate(asg, snapshot(world, {0: (0, 0)}, used={0: 150}), tasks)
    assert ok.ok and spent.by_rule() == {"energy": 1}


def test_same_altitude_violates_separation_layers_fix_it():
    world, tasks = make_world_tasks(n_cells=25)
    snap, asg = _plan(world, tasks, {0: (0, 0), 1: (5, 0), 2: (10, 0)})
    same = PlanValidator().validate(asg, snap, tasks, {0: 40.0, 1: 40.0, 2: 40.0})
    layered = PlanValidator().validate(asg, snap, tasks, {0: 40.0, 1: 45.0, 2: 50.0})
    assert same.by_rule().get("separation", 0) > 0
    assert "separation" not in layered.by_rule()


def test_altitude_corridor():
    world, tasks = make_world_tasks(n_cells=4)
    snap, asg = _plan(world, tasks, {0: (0, 0), 1: (1000, 1000)})
    rep = PlanValidator(z_range=(10, 120)).validate(asg, snap, tasks, {0: 40.0, 1: 150.0})
    assert rep.by_rule() == {"altitude": 1}


def test_path_avoids_nofly_zone():
    """Путь строится с обходом зоны, поэтому даже перелёт через неё не нарушает geozone."""
    world, tasks = make_world_tasks(n_cells=25, nofly=[ZONE])
    path = planned_path(world, (300, 500), [], (700, 500))
    assert len(path) > 2  # обходные точки
    snap, asg = _plan(world, tasks)
    rep = PlanValidator().validate(asg, snap, tasks, LAYERS)
    assert "geozone" not in rep.by_rule(), rep.describe()


def test_zone_inside_task_is_reported():
    """Зону объявили после построения задач: точки галсов внутри неё ловятся валидатором."""
    world, tasks = make_world_tasks(n_cells=4)  # ячейки 500×500: концы галсов на x=500 внутри зоны
    world.nofly.append(ZONE)
    asg = Assignment(plans={0: [t.id for t in tasks]})
    rep = PlanValidator().validate(asg, snapshot(world, {0: (0, 0)}), tasks)
    assert rep.by_rule().get("geozone", 0) >= 1
