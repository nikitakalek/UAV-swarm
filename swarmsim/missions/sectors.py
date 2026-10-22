"""Произвольные сектора, заданные пользователем (M2-13).

Формат в миссии (``sectors_version: 1``)::

    mission:
      sectors_version: 1
      sector_assignment: centroid   # centroid | overlap | clip
      sectors:
        - id: north
          polygon: [[0, 500], [1000, 500], [1000, 1000], [0, 1000]]
          agents: [0, 1]            # необязательно

* ``centroid`` — задача сетки принадлежит сектору, в котором лежит её центроид;
* ``overlap``  — сектору с наибольшим перекрытием (оценка по выборке точек);
* ``clip``     — ячейки сетки обрезаются по секторам, задачи целиком внутри сектора.

Диаграмма Вороного тут не нужна: назначение — по попаданию в полигон.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

from swarmsim import geometry as geo
from swarmsim.interfaces import Assignment, SwarmSnapshot
from swarmsim.missions.base import Task, TaskSet

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.scenario import Scenario, SectorConfig

SECTORS_VERSION = 1


# --------------------------------------------------------------------------- валидация
def validate_sectors(sc: "Scenario", samples: int = 80) -> None:
    m = sc.mission
    if m.sectors_version != SECTORS_VERSION:
        raise ValueError(f"mission.sectors_version={m.sectors_version} не поддерживается ({SECTORS_VERSION})")
    area = list(m.area) if m.area else sc.world.bounds_polygon()
    nofly = sc.world.nofly
    problems: list[str] = []
    ids = [s.id for s in m.sectors]
    if len(set(ids)) != len(ids):
        problems.append("id секторов не уникальны")
    for s in m.sectors:
        if not geo.polygon_inside(s.polygon, area):
            problems.append(f"сектор {s.id} выходит за границы области")
        for k, z in enumerate(nofly):
            if geo.polygons_overlap(s.polygon, z):
                problems.append(f"сектор {s.id} заходит в бесполётную зону {k}")
        for a in s.agents:
            if not 0 <= a < sc.agents.count:
                problems.append(f"сектор {s.id}: агента {a} нет")
    for i in range(len(m.sectors)):
        for j in range(i + 1, len(m.sectors)):
            if geo.polygons_overlap(m.sectors[i].polygon, m.sectors[j].polygon):
                problems.append(f"сектора {m.sectors[i].id} и {m.sectors[j].id} пересекаются")
    uncovered = _uncovered_points(area, nofly, [s.polygon for s in m.sectors], samples)
    if uncovered:
        x, y = uncovered[0]
        problems.append(f"сектора не покрывают область: {len(uncovered)} контрольных точек вне секторов, "
                        f"например ({x:.1f}, {y:.1f})")
    if problems:
        raise ValueError("ошибки разметки секторов: " + "; ".join(problems))


def _uncovered_points(area, nofly, sectors, samples: int) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = geo.bbox(area)
    out = []
    for i in range(samples):
        y = y0 + (i + 0.5) * (y1 - y0) / samples
        for j in range(samples):
            p = (x0 + (j + 0.5) * (x1 - x0) / samples, y)
            if not geo.point_in_polygon(p, area, boundary=False):
                continue
            if any(geo.point_in_polygon(p, z) for z in nofly):
                continue
            if not any(geo.point_in_polygon(p, s) for s in sectors):
                out.append(p)
    return out


# --------------------------------------------------------------------------- задачи → сектора
def assign_tasks_to_sectors(ts: TaskSet, sectors: Sequence["SectorConfig"], method: str = "centroid") -> None:
    for task in ts:
        task.sector = sector_of(task, sectors, method)


def sector_of(task: Task, sectors: Sequence["SectorConfig"], method: str = "centroid") -> str | None:
    if method == "overlap":
        pts = _sample_inside(task.polygon or task.waypoints, 12)
        best, best_n = None, 0
        for s in sectors:
            n = sum(geo.point_in_polygon(p, s.polygon) for p in pts)
            if n > best_n:
                best, best_n = s.id, n
        return best
    c = task.centroid
    for s in sectors:
        if geo.point_in_polygon(c, s.polygon):
            return s.id
    return None


def _sample_inside(poly: Sequence[tuple[float, float]], k: int) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = geo.bbox(poly)
    pts = [(x0 + (j + 0.5) * (x1 - x0) / k, y0 + (i + 0.5) * (y1 - y0) / k) for i in range(k) for j in range(k)]
    inside = [p for p in pts if len(poly) < 3 or geo.point_in_polygon(p, poly)]
    return inside or pts


def clip_cells_to_sectors(cells: Iterable, sectors: Sequence["SectorConfig"], swath: float, nofly,
                          min_area: float) -> list[Task]:
    """Режим ``clip``: ячейка сетки ∩ сектор → отдельная задача внутри сектора."""
    from swarmsim.planning.coverage import lawnmower

    tasks = []
    for r, c, cell in cells:
        for s in sectors:
            poly = geo.clip_polygon(s.polygon, cell)
            if len(poly) < 3 or geo.area(poly) < min_area:
                continue
            wps = lawnmower(cell, s.polygon, swath, nofly)
            if len(wps) < 2:
                continue
            tasks.append(Task(id=f"{s.id}_r{r:02d}_c{c:02d}", kind="sweep", waypoints=wps, polygon=poly,
                              sector=s.id, meta={"row": r, "col": c, "area_m2": geo.area(poly)}))
    return tasks


# --------------------------------------------------------------------------- распределитель
class ManualSectorAllocator:
    """Назначение по ручной разметке: задачи сектора — агентам этого сектора.

    Сектора без указанных агентов раздаются по кругу свободным агентам. При отказе
    задачи уходят живым агентам того же сектора, а если их нет — наименее
    загруженному живому агенту.
    """

    name = "manual_sectors"
    requires_gcs = True

    def __init__(self, scenario: "Scenario" = None, **params) -> None:
        self.sectors = list(scenario.mission.sectors) if scenario is not None else []
        self.params = params

    def should_trigger(self, t, tasks, pending) -> bool:
        return bool(pending) or bool(tasks.unassigned())

    def _owners(self, alive: list[int]) -> dict[str, list[int]]:
        bound = {a for s in self.sectors for a in s.agents}
        free = [a for a in alive if a not in bound]
        owners: dict[str, list[int]] = {}
        k = 0
        for s in self.sectors:
            if s.agents:
                owners[s.id] = [a for a in s.agents if a in alive]
            elif free:
                owners[s.id] = [free[k % len(free)]]
                k += 1
            else:
                owners[s.id] = []
        return owners

    def allocate(self, state: SwarmSnapshot, tasks: TaskSet) -> Assignment:
        alive = state.alive_ids()
        plans: dict[int, list[str]] = {i: [] for i in alive}
        reasons: dict[str, str] = {}
        if not alive:
            return Assignment(plans=plans)
        owned: set[str] = set()
        for i in alive:
            v = state.agents[i]
            for tid in ([v.current_task] if v.current_task else []) + v.queue:
                if tid in tasks and not tasks[tid].is_finished:
                    plans[i].append(tid)
                    owned.add(tid)
        owners = self._owners(alive)
        rr: dict[str, int] = {}
        for task in tasks.unfinished():
            if task.id in owned:
                continue
            cands = owners.get(task.sector or "", [])
            if cands:
                k = rr.get(task.sector, 0)
                aid = min(cands, key=lambda i: (len(plans[i]), i)) if task.t_assigned else cands[k % len(cands)]
                rr[task.sector] = k + 1
                reasons[task.id] = f"попадание в сектор {task.sector} → агент {aid}"
            else:
                aid = min(alive, key=lambda i: (len(plans[i]), i))
                reasons[task.id] = (f"у сектора {task.sector} нет живых агентов → наименее загруженный {aid}"
                                    if task.sector else f"задача вне секторов → наименее загруженный {aid}")
            plans[aid].append(task.id)
        return Assignment(plans=plans, reasons=reasons)
