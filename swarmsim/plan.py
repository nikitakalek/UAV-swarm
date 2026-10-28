"""Планирование и проверка маршрутов (Модуль 1, задачи M1-3 и M1-4).

**Маршрутизация (M1-3).** Упорядочивает назначенные агенту точки так, чтобы маршрут
«старт → задачи → домой» был коротким, и проверяет, влезает ли он в энергию.
Алгоритм — варианта A: **ближайший сосед + 2-opt**. Детерминирован, без внешних
зависимостей. Энергия **внедряется**: ``energy_fn(waypoints, n_service_stops) -> float``.

**Проверка плана (M1-4).** ``PlanValidator`` отвечает на вопрос «можно ли это исполнять»:
геозоны, энергия, сепарация между агентами, высотный коридор. Недопустимый план
возвращается со списком конкретных нарушений — ничего не «подчищается» молча.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from .alloc import AgentId, Assignment, TaskLike

Point2D = tuple[float, float]
# (путевые точки, число остановок на задачи) -> энергия
EnergyFn = Callable[[Sequence[Point2D], int], float]


# ---------------------------------------------------------------------------
# Модель энергии (заглушка до M3-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleEnergy:
    """Простая модель энергии в джоулях: крейсер + висение на задачу.

    Это не физика, а оценка. Реальную модель (формула Zeng et al.) даст M3-4;
    маршрутизатору достаточно, чтобы функция была монотонна по длине и числу задач.
    """

    cruise_speed: float = 10.0  # м/с
    power_cruise: float = 200.0  # Вт в крейсерском полёте
    power_service: float = 250.0  # Вт при обслуживании задачи
    service_time: float = 5.0  # с на задачу (ячейка ~40 м при 10 м/с)

    def __call__(self, waypoints: Sequence[Point2D], n_service_stops: int) -> float:
        return (
            self.power_cruise * polyline_length(waypoints) / self.cruise_speed
            + self.power_service * self.service_time * n_service_stops
        )


default_energy: EnergyFn = SimpleEnergy()


# ---------------------------------------------------------------------------
# Результат маршрутизации
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """Маршрут одного агента."""

    agent_id: AgentId
    task_ids: tuple[int, ...]  # порядок обхода задач
    waypoints: tuple[Point2D, ...]  # старт → задачи → домой
    length: float  # длина полилинии, м
    energy: float  # оценка энергии, Дж
    feasible: bool  # влезает ли в бюджет с резервом
    reason: str = ""  # причина недопустимости

    @property
    def n_tasks(self) -> int:
        return len(self.task_ids)


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------


def _dist(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(points: Sequence[Point2D]) -> float:
    return sum(_dist(a, b) for a, b in zip(points, points[1:]))


# ---------------------------------------------------------------------------
# Алгоритм: ближайший сосед + 2-opt
# ---------------------------------------------------------------------------


def _nearest_neighbour(points: Sequence[Point2D], start: Point2D) -> list[int]:
    """Порядок обхода «сначала ближайший». При равных расстояниях — меньший индекс."""
    remaining = list(range(len(points)))
    order: list[int] = []
    current = start
    while remaining:
        best = min(remaining, key=lambda i: (_dist(current, points[i]), i))
        order.append(best)
        remaining.remove(best)
        current = points[best]
    return order


def _two_opt(
    points: Sequence[Point2D],
    order: Sequence[int],
    start: Point2D,
    home: Point2D | None,
    max_passes: int = 100,
) -> list[int]:
    """Улучшает порядок обхода, разворачивая участки пути (2-opt).

    Узел ``-1`` — старт (зафиксирован), ``-2`` — домой (зафиксирован, если задан).
    """
    nodes = [-1] + list(order) + ([-2] if home is not None else [])

    def coord(node: int) -> Point2D:
        if node == -1:
            return start
        if node == -2:
            return home  # type: ignore[return-value]
        return points[node]

    n = len(nodes)
    end = n - 2 if home is not None else n - 1

    for _ in range(max_passes):
        improved = False
        for i in range(1, end + 1):
            for k in range(i + 1, end + 1):
                prev, a, b = coord(nodes[i - 1]), coord(nodes[i]), coord(nodes[k])
                has_next = k + 1 < n
                nxt = coord(nodes[k + 1]) if has_next else None
                old = _dist(prev, a) + (_dist(b, nxt) if nxt is not None else 0.0)
                new = _dist(prev, b) + (_dist(a, nxt) if nxt is not None else 0.0)
                if new < old - 1e-12:
                    nodes[i : k + 1] = reversed(nodes[i : k + 1])
                    improved = True
        if not improved:
            break

    return [node for node in nodes if node >= 0]


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def route_tasks(
    tasks: Sequence[TaskLike],
    start: Point2D,
    home: Point2D | None = None,
    *,
    energy_fn: EnergyFn | None = None,
    budget: float | None = None,
    reserve: float = 0.2,
    agent_id: AgentId = 0,
) -> Route:
    """Строит маршрут агента по его задачам.

    :param start: текущая позиция агента.
    :param home: база, куда нужно вернуться (``None`` — без возврата).
    :param energy_fn: модель энергии; по умолчанию :class:`SimpleEnergy`.
    :param budget: доступная энергия, Дж; ``None`` — считать бюджет неограниченным.
    :param reserve: обязательный резерв (доля), по умолчанию 20 %.
    """
    energy_fn = energy_fn or default_energy
    points = [tuple(t.centroid) for t in tasks]

    if not points:
        waypoints = (start,) if home is None or home == start else (start, home)
        energy = energy_fn(waypoints, 0)
        feasible, reason = _check_budget(energy, budget, reserve)
        return Route(agent_id, (), tuple(waypoints), polyline_length(waypoints), energy, feasible, reason)

    order = _nearest_neighbour(points, start)
    order = _two_opt(points, order, start, home)

    waypoints = [start]
    waypoints.extend(points[i] for i in order)
    if home is not None:
        waypoints.append(home)

    length = polyline_length(waypoints)
    energy = energy_fn(waypoints, len(points))
    feasible, reason = _check_budget(energy, budget, reserve)

    return Route(
        agent_id=agent_id,
        task_ids=tuple(int(tasks[i].id) for i in order),
        waypoints=tuple(waypoints),
        length=length,
        energy=energy,
        feasible=feasible,
        reason=reason,
    )


def route_assignment(
    assignment: Assignment,
    tasks_by_id: Mapping[int, TaskLike],
    positions: Mapping[AgentId, Point2D],
    *,
    homes: Mapping[AgentId, Point2D] | None = None,
    budgets: Mapping[AgentId, float] | None = None,
    energy_fn: EnergyFn | None = None,
    reserve: float = 0.2,
) -> dict[AgentId, Route]:
    """Маршрутизирует все назначения: ``{agent_id: Route}``.

    ``positions``/``homes``/``budgets`` заполняет Модуль 2 (сейчас — заглушки).
    """
    routes: dict[AgentId, Route] = {}
    for agent_id in assignment.agents():
        tasks = [tasks_by_id[t] for t in assignment.tasks_of(agent_id)]
        routes[agent_id] = route_tasks(
            tasks,
            start=positions[agent_id],
            home=(homes or {}).get(agent_id),
            energy_fn=energy_fn,
            budget=(budgets or {}).get(agent_id),
            reserve=reserve,
            agent_id=agent_id,
        )
    return routes


def _check_budget(
    energy: float, budget: float | None, reserve: float
) -> tuple[bool, str]:
    if budget is None:
        return True, ""
    limit = budget * (1.0 - reserve)
    if energy <= limit:
        return True, ""
    return False, (
        f"энергия {energy:.0f} Дж превышает лимит {limit:.0f} Дж "
        f"(бюджет {budget:.0f}, резерв {reserve:.0%})"
    )


# ---------------------------------------------------------------------------
# Проверка плана (M1-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """Одно нарушение правил плана."""

    rule: str  # "geozone" | "energy" | "separation" | "altitude" | "domain"
    agents: tuple[AgentId, ...]
    detail: str
    location: Point2D | None = None

    def __str__(self) -> str:
        who = ", ".join(str(a) for a in self.agents)
        loc = f" @ ({self.location[0]:.0f}, {self.location[1]:.0f})" if self.location else ""
        return f"{self.rule} [агенты {who}]: {self.detail}{loc}"


@dataclass(frozen=True)
class ValidationReport:
    """Результат проверки плана."""

    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    def by_rule(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v.rule] = counts.get(v.rule, 0) + 1
        return counts

    def describe(self) -> str:
        if self.ok:
            return "план допустим: нарушений нет"
        lines = [f"план НЕДОПУСТИМ: нарушений {len(self.violations)}"]
        lines.extend(f"  - {v}" for v in self.violations)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


@dataclass
class PlanValidator:
    """Проверяет план (маршруты всех агентов) перед исполнением.

    Четыре жёстких правила — геозоны, энергия, сепарация, высотный коридор.
    Нарушение любого означает, что план нельзя исполнять; нарушения не
    «исправляются», а возвращаются списком.

    :param no_fly: бесполётные зоны (полигоны shapely).
    :param domain: граница области; ``None`` — не проверять.
    :param min_sep: минимальное горизонтальное расстояние между траекториями, м.
    :param min_sep_vertical: минимальное вертикальное расхождение, м: если два
        агента разнесены по высоте на эту величину, горизонтальная сепарация
        для них не требуется.
    :param z_range: допустимый коридор высот, м; ``None`` — не проверять.
    :param reserve: обязательный резерв энергии (доля).
    """

    no_fly: tuple[BaseGeometry, ...] = ()
    domain: BaseGeometry | None = None
    min_sep: float = 5.0
    min_sep_vertical: float = 3.0
    z_range: tuple[float, float] | None = (0.0, 120.0)
    reserve: float = 0.2
    tolerance: float = 1e-6

    def validate(
        self,
        routes: Mapping[AgentId, Route],
        *,
        budgets: Mapping[AgentId, float] | None = None,
        altitudes: Mapping[AgentId, float] | None = None,
    ) -> ValidationReport:
        violations: list[Violation] = []
        lines: dict[AgentId, LineString] = {}

        for agent, route in routes.items():
            if len(route.waypoints) >= 2:
                lines[agent] = LineString(route.waypoints)

            self._check_energy(agent, route, budgets, violations)

            line = lines.get(agent)
            if line is None:
                continue
            self._check_geozones(agent, line, violations)
            self._check_domain(agent, line, violations)

        self._check_altitude(routes, altitudes, violations)
        self._check_separation(lines, altitudes, violations)

        return ValidationReport(tuple(violations))

    # -- отдельные правила --------------------------------------------------

    def _check_energy(self, agent, route, budgets, violations) -> None:
        if budgets is not None and agent in budgets:
            limit = budgets[agent] * (1.0 - self.reserve)
            if route.energy > limit:
                violations.append(
                    Violation(
                        "energy",
                        (agent,),
                        f"энергия {route.energy:.0f} Дж > лимита {limit:.0f} Дж "
                        f"(резерв {self.reserve:.0%})",
                    )
                )
        elif not route.feasible:
            violations.append(
                Violation("energy", (agent,), route.reason or "маршрут недопустим по энергии")
            )

    def _check_geozones(self, agent, line: LineString, violations) -> None:
        for index, zone in enumerate(self.no_fly):
            if line.intersects(zone):
                location = _point_of(line.intersection(zone))
                violations.append(
                    Violation(
                        "geozone",
                        (agent,),
                        f"маршрут пересекает бесполётную зону #{index}",
                        location,
                    )
                )

    def _check_domain(self, agent, line: LineString, violations) -> None:
        if self.domain is None:
            return
        outside = line.difference(self.domain)
        if not outside.is_empty and outside.length > self.tolerance:
            violations.append(
                Violation(
                    "domain",
                    (agent,),
                    f"маршрут выходит за пределы области на {outside.length:.1f} м",
                    _point_of(outside),
                )
            )

    def _check_altitude(self, routes, altitudes, violations) -> None:
        if altitudes is None or self.z_range is None:
            return
        z_min, z_max = self.z_range
        for agent in routes:
            z = altitudes.get(agent)
            if z is not None and not (z_min <= z <= z_max):
                violations.append(
                    Violation(
                        "altitude",
                        (agent,),
                        f"высота {z:.1f} м вне коридора [{z_min:.0f}, {z_max:.0f}] м",
                    )
                )

    def _check_separation(self, lines, altitudes, violations) -> None:
        agents = list(lines)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                if self._vertically_separated(a, b, altitudes):
                    continue
                la, lb = lines[a], lines[b]
                if _bbox_gap(la.bounds, lb.bounds) >= self.min_sep:
                    continue
                distance = la.distance(lb)
                if distance < self.min_sep:
                    p, q = nearest_points(la, lb)
                    violations.append(
                        Violation(
                            "separation",
                            (a, b),
                            f"расстояние {distance:.1f} м < {self.min_sep:.1f} м",
                            ((p.x + q.x) / 2.0, (p.y + q.y) / 2.0),
                        )
                    )

    def _vertically_separated(self, a, b, altitudes) -> bool:
        if altitudes is None:
            return False
        za, zb = altitudes.get(a), altitudes.get(b)
        if za is None or zb is None:
            return False
        return abs(za - zb) >= self.min_sep_vertical


def _point_of(geometry: BaseGeometry | None) -> Point2D | None:
    """Точка-представитель геометрии для отчёта о нарушении."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "Point":
        return (float(geometry.x), float(geometry.y))
    centroid = geometry.centroid
    return (float(centroid.x), float(centroid.y))


def _bbox_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Расстояние между габаритными прямоугольниками (0, если пересекаются)."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)
