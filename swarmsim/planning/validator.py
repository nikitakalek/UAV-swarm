"""PlanValidator — проверка плана группы перед исполнением (М1).

Валидатор превращает «надеемся, что планировщик не ошибся» в «недопустимый план не
будет исполнен». Он строит путь каждого аппарата так же, как его полетит ``Agent``
(текущая позиция → обход бесполётных зон → галсы задач → домой) и проверяет правила:

* ``assignment`` — каждая незавершённая задача назначена ровно одному живому аппарату;
* ``geozone`` — путь не пересекает бесполётные зоны;
* ``domain`` — путь не выходит за границу области (с допуском ``domain_margin``);
* ``energy`` — путь укладывается в остаток батареи с обязательным резервом (20 %);
* ``altitude`` — высота в разрешённом коридоре;
* ``separation`` — аппараты на одной высоте (разнос < ``min_sep_vertical``) не сближают
  пути ближе ``min_sep`` по горизонтали. Эшелонирование по высоте снимает это правило.

Нарушения не исправляются, а возвращаются списком (``ValidationReport``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

from swarmsim.interfaces import Assignment, SwarmSnapshot

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.missions.base import TaskSet
    from swarmsim.world import World

Point2D = tuple[float, float]


@dataclass(frozen=True)
class Violation:
    rule: str  # assignment | geozone | domain | energy | altitude | separation
    agents: tuple[int, ...]
    detail: str
    location: Optional[Point2D] = None

    def __str__(self) -> str:
        who = ", ".join(str(a) for a in self.agents)
        loc = f" @ ({self.location[0]:.0f}, {self.location[1]:.0f})" if self.location else ""
        return f"{self.rule} [аппараты {who}]: {self.detail}{loc}"


@dataclass(frozen=True)
class ValidationReport:
    violations: tuple[Violation, ...] = ()
    paths: Mapping[int, tuple[Point2D, ...]] = field(default_factory=dict, compare=False)

    @property
    def ok(self) -> bool:
        return not self.violations

    def by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.rule] = out.get(v.rule, 0) + 1
        return out

    def describe(self) -> str:
        if self.ok:
            return "план допустим: нарушений нет"
        return "\n".join([f"план НЕДОПУСТИМ: нарушений {len(self.violations)}"] + [f"  - {v}" for v in self.violations])

    __str__ = describe


@dataclass
class PlanValidator:
    """Проверка плана. Параметры — ограничения аппарата и правил полётов.

    :param battery_wh: ёмкость батареи, Вт·ч (``None`` — энергию не проверять).
    :param power_w: средняя мощность в полёте, Вт (оценка расхода: P · длина / v).
    :param reserve: обязательный резерв энергии, доля.
    :param min_sep: минимальное горизонтальное расстояние между путями, м.
    :param min_sep_vertical: разнос по высоте, при котором горизонтальная сепарация не нужна, м.
    :param z_range: коридор высот, м.
    :param domain_margin: допуск выхода за границу области, м (галс «змейки» у наклонной
        границы может выйти на полширины полосы — известное свойство покрытия).
    """

    battery_wh: Optional[float] = None
    power_w: float = 250.0
    reserve: float = 0.2
    min_sep: float = 5.0
    min_sep_vertical: float = 3.0
    z_range: tuple[float, float] = (0.0, 120.0)
    domain_margin: float = 0.0

    def validate(self, assignment: Assignment, state: SwarmSnapshot, tasks: "TaskSet",
                 altitudes: Optional[Mapping[int, float]] = None) -> ValidationReport:
        world = state.world
        out: list[Violation] = []
        self._check_assignment(assignment, state, tasks, out)

        paths: dict[int, tuple[Point2D, ...]] = {}
        for aid, plan in assignment.plans.items():
            if aid not in state.agents:
                continue
            v = state.agents[aid]
            path = planned_path(world, tuple(v.pos), [tasks[t] for t in plan if t in tasks], tuple(v.home))
            paths[aid] = path
            line = LineString(path) if len(path) >= 2 else None
            if line is not None:
                self._check_geozones(aid, line, world, out)
                self._check_domain(aid, line, world, out)
                self._check_energy(aid, line.length, v.speed, v.energy_used_wh, out)

        if altitudes:
            self._check_altitude(altitudes, assignment, out)
        self._check_separation(paths, altitudes or {}, out)
        return ValidationReport(tuple(out), paths)

    # ------------------------------------------------------------------ правила
    def _check_assignment(self, asg: Assignment, state: SwarmSnapshot, tasks: "TaskSet", out: list) -> None:
        seen: dict[str, int] = {}
        for aid in sorted(asg.plans):
            alive = aid in state.agents and state.agents[aid].alive
            for tid in asg.plans[aid]:
                if tid not in tasks:
                    out.append(Violation("assignment", (aid,), f"неизвестная задача {tid}"))
                    continue
                if tid in seen:
                    out.append(Violation("assignment", (seen[tid], aid), f"задача {tid} назначена дважды"))
                seen[tid] = aid
                if not alive:
                    out.append(Violation("assignment", (aid,), f"задача {tid} назначена выбывшему аппарату"))
        missing = [t.id for t in tasks.unfinished() if t.id not in seen]
        if missing:
            out.append(Violation("assignment", (), f"без исполнителя: {', '.join(missing)}"))

    def _check_geozones(self, aid: int, line: LineString, world: "World", out: list) -> None:
        for k, zone in enumerate(world.nofly):
            # проход по границе зоны (галсы обрезаны по ней, обход по углам рамки) — не нарушение
            inside = Polygon(zone).buffer(-0.01)
            inter = line.intersection(inside)
            if not inter.is_empty and inter.length > 1e-6:
                out.append(Violation("geozone", (aid,), f"путь проходит через бесполётную зону #{k}", _point_of(inter)))

    def _check_domain(self, aid: int, line: LineString, world: "World", out: list) -> None:
        dom = Polygon(world.bounds).buffer(self.domain_margin + 1e-6)
        outside = line.difference(dom)
        if not outside.is_empty and outside.length > 1e-6:
            out.append(Violation("domain", (aid,), f"путь выходит за границу области на {outside.length:.1f} м",
                                 _point_of(outside)))

    def _check_energy(self, aid: int, length: float, speed: float, used_wh: float, out: list) -> None:
        if self.battery_wh is None:
            return
        need = self.power_w * (length / max(speed, 1e-9)) / 3600.0
        limit = (self.battery_wh - used_wh) - self.reserve * self.battery_wh
        if need > limit:
            out.append(Violation("energy", (aid,), f"нужно {need:.1f} Вт·ч, доступно {max(limit, 0):.1f} Вт·ч "
                                                   f"(резерв {self.reserve:.0%} от {self.battery_wh:.0f})"))

    def _check_altitude(self, altitudes: Mapping[int, float], asg: Assignment, out: list) -> None:
        lo, hi = self.z_range
        for aid in sorted(asg.plans):
            z = altitudes.get(aid)
            if z is not None and not (lo <= z <= hi):
                out.append(Violation("altitude", (aid,), f"высота {z:.1f} м вне коридора [{lo:.0f}, {hi:.0f}] м"))

    def _check_separation(self, paths: Mapping[int, Sequence[Point2D]], altitudes: Mapping[int, float],
                          out: list) -> None:
        ids = sorted(a for a, p in paths.items() if len(p) >= 2)
        lines = {a: LineString(paths[a]) for a in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                za, zb = altitudes.get(a), altitudes.get(b)
                if za is not None and zb is not None and abs(za - zb) >= self.min_sep_vertical:
                    continue
                d = lines[a].distance(lines[b])
                if d < self.min_sep:
                    p, q = nearest_points(lines[a], lines[b])
                    out.append(Violation("separation", (a, b), f"пути сближаются до {d:.1f} м < {self.min_sep:.1f} м "
                                                               f"на одной высоте", ((p.x + q.x) / 2, (p.y + q.y) / 2)))


def planned_path(world: "World", start: Point2D, plan: Sequence, home: Optional[Point2D]) -> tuple[Point2D, ...]:
    """Путь аппарата так, как его полетит ``Agent``: обход зон, галсы задач, возврат домой."""
    path: list[Point2D] = [start]
    pos = start
    for task in plan:
        if task.is_finished:
            continue
        for wp in task.waypoints[task.progress:]:
            path.extend(world.free_path(pos, tuple(wp)))
            pos = tuple(wp)
    if home is not None:
        path.extend(world.free_path(pos, home))
    return tuple(path)


def _point_of(geom: BaseGeometry) -> Optional[Point2D]:
    if geom is None or geom.is_empty:
        return None
    c = geom if isinstance(geom, Point) else geom.centroid
    return (float(c.x), float(c.y))
