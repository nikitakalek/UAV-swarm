"""Тесты маршрутизации агента (M1-3)."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import pytest
from shapely.geometry import box

from swarmsim.alloc import Assignment
from swarmsim.plan import (
    PlanValidator,
    Route,
    ValidationReport,
    polyline_length,
    route_assignment,
    route_tasks,
)


@dataclass(frozen=True)
class StubTask:
    id: int
    centroid: tuple[float, float]


def tasks_line(n: int, step: float = 10.0) -> list[StubTask]:
    return [StubTask(i, (step * (i + 1), 0.0)) for i in range(n)]


# --- корректность маршрута -------------------------------------------------


def test_visits_every_task_once():
    tasks = tasks_line(6)
    route = route_tasks(tasks, start=(0.0, 0.0), home=(0.0, 0.0))
    assert sorted(route.task_ids) == [t.id for t in tasks]
    assert route.n_tasks == 6


def test_starts_at_start_and_returns_home():
    tasks = tasks_line(5)
    route = route_tasks(tasks, start=(0.0, 0.0), home=(100.0, 100.0))
    assert route.waypoints[0] == (0.0, 0.0)
    assert route.waypoints[-1] == (100.0, 100.0)
    assert len(route.waypoints) == len(tasks) + 2


def test_without_home_ends_at_last_task():
    tasks = tasks_line(4)
    route = route_tasks(tasks, start=(0.0, 0.0), home=None)
    assert route.waypoints[0] == (0.0, 0.0)
    assert len(route.waypoints) == len(tasks) + 1


def test_length_matches_waypoints():
    tasks = tasks_line(5)
    route = route_tasks(tasks, start=(0.0, 0.0), home=(0.0, 0.0))
    assert route.length == pytest.approx(polyline_length(route.waypoints))


def test_points_on_a_line_are_visited_in_order():
    """Для точек на прямой кратчайший путь монотонен — заодно проверка 2-opt."""
    pts = [(30.0, 0.0), (10.0, 0.0), (50.0, 0.0), (20.0, 0.0)]
    tasks = [StubTask(i, p) for i, p in enumerate(pts)]
    route = route_tasks(tasks, start=(0.0, 0.0), home=(60.0, 0.0))
    assert route.task_ids == (1, 3, 0, 2)  # 10, 20, 30, 50


def test_single_task():
    route = route_tasks([StubTask(7, (10.0, 0.0))], start=(0.0, 0.0), home=(0.0, 0.0))
    assert route.task_ids == (7,)
    assert route.length == pytest.approx(20.0)


def test_no_tasks():
    route = route_tasks([], start=(5.0, 5.0), home=(5.0, 5.0))
    assert route.task_ids == ()
    assert route.waypoints == ((5.0, 5.0),)
    assert route.feasible


# --- 2-opt -----------------------------------------------------------------


def test_two_opt_is_local_optimum():
    """После улучшения не должно остаться ни одного выгодного разворота участка."""
    pts = [(10.0, 50.0), (20.0, -50.0), (30.0, 50.0), (40.0, -50.0), (50.0, 50.0), (60.0, -50.0)]
    tasks = [StubTask(i, p) for i, p in enumerate(pts)]
    route = route_tasks(tasks, start=(0.0, 0.0), home=(100.0, 0.0))

    nodes = [route.waypoints[0]] + [bp for bp in route.waypoints[1:-1]] + [route.waypoints[-1]]
    n = len(nodes)

    def d(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])

    for i in range(1, n - 1):
        for k in range(i + 1, n - 1):
            prev, a, b, nxt = nodes[i - 1], nodes[i], nodes[k], nodes[k + 1]
            old = d(prev, a) + d(b, nxt)
            new = d(prev, b) + d(a, nxt)
            assert new >= old - 1e-9, "остался выгодный разворот — 2-opt не досчитал"


def test_determinism():
    tasks = tasks_line(7)
    r1 = route_tasks(tasks, start=(0.0, 0.0), home=(0.0, 0.0))
    r2 = route_tasks(tasks, start=(0.0, 0.0), home=(0.0, 0.0))
    assert r1 == r2


# --- энергия и бюджет ------------------------------------------------------


def test_energy_grows_with_tasks():
    short = route_tasks(tasks_line(2), start=(0.0, 0.0), home=(0.0, 0.0))
    long = route_tasks(tasks_line(8), start=(0.0, 0.0), home=(0.0, 0.0))
    assert long.energy > short.energy


def test_no_budget_is_always_feasible():
    route = route_tasks(tasks_line(10), start=(0.0, 0.0), home=(0.0, 0.0), budget=None)
    assert route.feasible
    assert route.reason == ""


def test_budget_exceeded_marks_infeasible():
    route = route_tasks(tasks_line(10), start=(0.0, 0.0), home=(0.0, 0.0), budget=1000.0)
    assert not route.feasible
    assert "превышает" in route.reason


def test_reserve_is_applied():
    tasks = tasks_line(6)
    enough = route_tasks(tasks, start=(0.0, 0.0), home=(0.0, 0.0), budget=1e9)
    assert enough.feasible

    # ровно в бюджет — но с обязательным резервом 20 % уже не влезает
    exact = route_tasks(tasks, start=(0.0, 0.0), home=(0.0, 0.0), budget=enough.energy)
    assert not exact.feasible

    # запас 1/0.8 снимает проблему
    margin = route_tasks(
        tasks, start=(0.0, 0.0), home=(0.0, 0.0), budget=enough.energy / 0.8 + 1.0
    )
    assert margin.feasible


# --- маршрутизация всего назначения ----------------------------------------


def test_route_assignment():
    tasks = {i: StubTask(i, (10.0 * i + 5.0, 0.0)) for i in range(6)}
    assignment = Assignment({0: (0, 1, 2), 1: (3, 4, 5)})
    routes = route_assignment(
        assignment,
        tasks,
        positions={0: (0.0, 0.0), 1: (100.0, 0.0)},
        homes={0: (0.0, 0.0), 1: (100.0, 0.0)},
    )

    assert set(routes) == {0, 1}
    assert sorted(routes[0].task_ids) == [0, 1, 2]
    assert sorted(routes[1].task_ids) == [3, 4, 5]
    assert all(isinstance(r, Route) for r in routes.values())


def test_route_assignment_with_budgets():
    tasks = {i: StubTask(i, (10.0 * i + 5.0, 0.0)) for i in range(6)}
    assignment = Assignment({0: (0, 1, 2), 1: (3, 4, 5)})
    routes = route_assignment(
        assignment,
        tasks,
        positions={0: (0.0, 0.0), 1: (100.0, 0.0)},
        budgets={0: 1e9, 1: 1.0},
    )
    assert routes[0].feasible
    assert not routes[1].feasible


# --- проверка плана (M1-4) -------------------------------------------------


def make_route(agent_id, waypoints, *, energy=100.0, feasible=True):
    wps = tuple(waypoints)
    return Route(
        agent_id=agent_id,
        task_ids=(),
        waypoints=wps,
        length=polyline_length(wps),
        energy=energy,
        feasible=feasible,
        reason="" if feasible else "не хватает энергии",
    )


def test_valid_plan_passes():
    routes = {
        0: make_route(0, [(0, 0), (10, 0)]),
        1: make_route(1, [(0, 100), (10, 100)]),
    }
    report = PlanValidator(domain=box(0, 0, 200, 200)).validate(routes)
    assert isinstance(report, ValidationReport)
    assert report.ok
    assert "допустим" in report.describe()


def test_geozone_violation():
    routes = {0: make_route(0, [(0, 5), (10, 5)])}
    report = PlanValidator(no_fly=(box(4, 4, 6, 6),)).validate(routes)
    assert not report.ok
    assert report.by_rule() == {"geozone": 1}
    assert report.violations[0].location is not None


def test_geozone_clear():
    routes = {0: make_route(0, [(0, 0), (10, 0)])}
    assert PlanValidator(no_fly=(box(4, 4, 6, 6),)).validate(routes).ok


def test_energy_from_route_flag():
    routes = {0: make_route(0, [(0, 0), (10, 0)], feasible=False)}
    report = PlanValidator().validate(routes)
    assert report.by_rule() == {"energy": 1}
    assert "не хватает энергии" in report.violations[0].detail


def test_energy_from_budgets():
    routes = {0: make_route(0, [(0, 0), (10, 0)], energy=100.0)}
    report = PlanValidator(reserve=0.2).validate(routes, budgets={0: 100.0})
    assert report.by_rule() == {"energy": 1}


def test_energy_within_budget():
    routes = {0: make_route(0, [(0, 0), (10, 0)], energy=70.0)}
    assert PlanValidator(reserve=0.2).validate(routes, budgets={0: 100.0}).ok


def test_separation_violation():
    routes = {
        0: make_route(0, [(0, 0), (10, 0)]),
        1: make_route(1, [(0, 1), (10, 1)]),
    }
    report = PlanValidator(min_sep=5.0).validate(routes)
    assert report.by_rule() == {"separation": 1}
    assert set(report.violations[0].agents) == {0, 1}


def test_separation_far_apart():
    routes = {
        0: make_route(0, [(0, 0), (10, 0)]),
        1: make_route(1, [(0, 50), (10, 50)]),
    }
    assert PlanValidator(min_sep=5.0).validate(routes).ok


def test_vertical_separation_allows_close_routes():
    routes = {
        0: make_route(0, [(0, 0), (10, 0)]),
        1: make_route(1, [(0, 1), (10, 1)]),
    }
    report = PlanValidator(min_sep=5.0, min_sep_vertical=3.0).validate(
        routes, altitudes={0: 10.0, 1: 13.0}
    )
    assert report.ok


def test_altitude_corridor():
    routes = {0: make_route(0, [(0, 0), (10, 0)])}
    report = PlanValidator(z_range=(0.0, 120.0)).validate(routes, altitudes={0: 200.0})
    assert report.by_rule() == {"altitude": 1}


def test_domain_violation():
    routes = {0: make_route(0, [(0, 5), (20, 5)])}
    report = PlanValidator(domain=box(0, 0, 10, 10)).validate(routes)
    assert report.by_rule() == {"domain": 1}


def test_report_lists_violations():
    routes = {
        0: make_route(0, [(0, 5), (10, 5)]),
        1: make_route(1, [(0, 50), (10, 50)], feasible=False),
    }
    report = PlanValidator(no_fly=(box(4, 4, 6, 6),)).validate(routes)
    text = report.describe()
    assert "НЕДОПУСТИМ" in text
    assert "geozone" in text
    assert "energy" in text
    assert report.by_rule() == {"geozone": 1, "energy": 1}
