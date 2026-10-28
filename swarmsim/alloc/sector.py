"""SectorAllocator — распределение задач группы (М1).

* **Начальное распределение.** Рабочая область (объединение ячеек миссии) делится на
  N компактных секторов равной площади (``planning.partition``), секторы сопоставляются
  аппаратам по близости к их позициям, задачи достаются сектору по центроиду. Затем
  граничные ячейки перекладываются от самого загруженного аппарата к соседу, пока это
  уменьшает время завершения миссии (makespan).
* **Ремонт после отказа или новой задачи.** Живые аппараты сохраняют свои планы.
  Свободные задачи («сироты») раздаются по одной: каждую забирает аппарат, который с ней
  закончит раньше всех (ближайшая к концу его маршрута задача + текущая загрузка).
  Так недоделанная часть сектора делится между ближайшими и наименее загруженными.

Порядок задач в плане — маршрут (``planning.routing``): текущая задача остаётся первой.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any, Optional

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from swarmsim.interfaces import Assignment, SwarmSnapshot
from swarmsim.planning.partition import build_partition
from swarmsim.planning.routing import append_cost, order_tasks

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.missions.base import Task, TaskSet

# сопоставление «сектор → аппарат» перебором до этого N, дальше — жадно
_EXACT_MATCH_MAX = 7


class SectorAllocator:
    """Компактные секторы при старте, ремонт «кто закончит раньше» при событиях.

    :param repair: ``False`` — статический режим без перераспределения (для сравнения).
    :param balance: перекладывать граничные ячейки для выравнивания загрузки.
    """

    name = "sector"
    requires_gcs = False  # решение можно принять на борту, наземная станция не нужна

    def __init__(self, repair: bool = True, balance: bool = True, lattice: str = "grid",
                 rng: Any = None, scenario: Any = None, **_: Any) -> None:
        self.repair = repair
        self.balance = balance
        self.lattice = lattice
        self.last_sectors: dict[int, Any] = {}  # для визуализации: агент → shapely-сектор

    # ------------------------------------------------------------------ протокол Allocator
    def should_trigger(self, t: float, tasks: "TaskSet", pending: list) -> bool:
        return bool(pending) or bool(tasks.unassigned())

    def allocate(self, state: SwarmSnapshot, tasks: "TaskSet") -> Assignment:
        alive = state.alive_ids()
        if not alive:
            return Assignment(plans={})
        open_tasks = {t.id: t for t in tasks.unfinished()}

        current: dict[int, Optional[str]] = {}
        groups: dict[int, list[str]] = {}
        owned: set[str] = set()
        for i in alive:
            v = state.agents[i]
            cur = v.current_task if v.current_task in open_tasks else None
            current[i] = cur
            ids = [tid for tid in ([cur] if cur else []) + list(v.queue) if tid in open_tasks and tid not in owned]
            groups[i] = ids
            owned.update(ids)
        free = sorted(tid for tid in open_tasks if tid not in owned)

        reasons: dict[str, str] = {}
        mode = "keep"
        if free and not owned:
            mode = "initial"
            groups, reasons = self._initial(state, alive, [open_tasks[t] for t in free])
            if self.balance and len(alive) > 1:
                self._balance(state, groups, current, open_tasks, reasons)
        elif free and self.repair:
            mode = "repair"
            self._repair(state, groups, current, [open_tasks[t] for t in free], open_tasks, reasons)
            if self.balance and len(alive) > 1:
                self._balance(state, groups, current, open_tasks, reasons)

        plans: dict[int, list[str]] = {}
        loads: dict[int, float] = {}
        for i in alive:
            v = state.agents[i]
            est = order_tasks([open_tasks[t] for t in groups[i]], tuple(v.pos), tuple(v.home), first=current[i])
            plans[i] = list(est.order)
            loads[i] = round(est.length, 1)
        return Assignment(plans=plans, reasons=reasons, meta={"mode": mode, "load_m": loads})

    # ------------------------------------------------------------------ начальное распределение
    def _initial(self, state: SwarmSnapshot, alive: list[int],
                 free: list["Task"]) -> tuple[dict[int, list[str]], dict[str, str]]:
        groups: dict[int, list[str]] = {i: [] for i in alive}
        reasons: dict[str, str] = {}
        if len(alive) == 1:
            groups[alive[0]] = [t.id for t in free]
            reasons.update({t.id: f"единственный аппарат {alive[0]}" for t in free})
            self.last_sectors = {}
            return groups, reasons

        domain = _work_area(free)
        part = build_partition(domain, len(alive), lattice=self.lattice)
        sectors = list(part.sectors) + [Polygon()] * (len(alive) - len(part.sectors))
        owner = _match(sectors, [tuple(state.agents[i].pos) for i in alive])
        sector_of = {alive[a]: sectors[k] for k, a in enumerate(owner)}
        self.last_sectors = sector_of

        for t in free:
            p = Point(_centroid(t))
            aid = min(alive, key=lambda i: (sector_of[i].distance(p) if not sector_of[i].is_empty else math.inf, i))
            groups[aid].append(t.id)
            reasons[t.id] = f"сектор аппарата {aid}"
        return groups, reasons

    def _balance(self, state: SwarmSnapshot, groups: dict[int, list[str]], current: dict[int, Optional[str]],
                 tasks: dict[str, "Task"], reasons: dict[str, str], max_moves: int = 200) -> None:
        """Перекладывает граничные ячейки от самого загруженного аппарата к соседям.

        Загрузка — длина пути до конца последней задачи (так считается makespan);
        задача, которую аппарат уже выполняет, не перекладывается.
        """
        polys = {tid: _polygon(t) for tid, t in tasks.items()}
        fixed = {t for t in current.values() if t}

        def load(i: int, ids: list[str]) -> float:
            v = state.agents[i]
            return order_tasks([tasks[t] for t in ids], tuple(v.pos), None, first=current.get(i), max_passes=0).length

        loads = {i: load(i, ids) for i, ids in groups.items()}
        for _ in range(max_moves):
            a = max(loads, key=lambda i: (loads[i], -i))
            best: Optional[tuple[float, str, int, float, float]] = None
            for b in sorted(groups, key=lambda i: (loads[i], i)):
                if b == a or loads[b] >= loads[a]:
                    continue
                region = unary_union([polys[t] for t in groups[b]]) if groups[b] else None
                for tid in groups[a]:
                    if tid in fixed:
                        continue
                    if region is not None and region.distance(polys[tid]) > 1.0:
                        continue  # перекладываем только граничные ячейки — секторы остаются связными
                    la = load(a, [t for t in groups[a] if t != tid])
                    lb = load(b, groups[b] + [tid])
                    peak = max(la, lb)
                    if peak < loads[a] - 1e-6 and (best is None or (peak, tid, b) < best[:3]):
                        best = (peak, tid, b, la, lb)
            if best is None:
                return
            _, tid, b, la, lb = best
            groups[a].remove(tid)
            groups[b].append(tid)
            loads[a], loads[b] = la, lb
            reasons[tid] = f"сектор аппарата {a} → сосед {b}: выравнивание загрузки"

    # ------------------------------------------------------------------ ремонт
    def _repair(self, state: SwarmSnapshot, groups: dict[int, list[str]], current: dict[int, Optional[str]],
                free: list["Task"], tasks: dict[str, "Task"], reasons: dict[str, str]) -> None:
        load: dict[int, float] = {}
        end: dict[int, tuple[float, float]] = {}
        for i, ids in groups.items():
            v = state.agents[i]
            est = order_tasks([tasks[t] for t in ids], tuple(v.pos), None, first=current[i])
            load[i] = est.length
            end[i] = tuple(tasks[est.order[-1]].waypoints[-1]) if est.order else tuple(v.pos)

        pool = sorted(free, key=lambda t: t.id)
        while pool:
            best: Optional[tuple[float, int, int, float, tuple[float, float]]] = None
            for i in sorted(groups):
                k = min(range(len(pool)), key=lambda j: (math.dist(end[i], tuple(pool[j].location)), pool[j].id))
                add, new_end = append_cost(end[i], pool[k])
                finish = load[i] + add
                if best is None or (finish, i) < (best[0], best[1]):
                    best = (finish, i, k, add, new_end)
            finish, i, k, add, new_end = best
            task = pool.pop(k)
            groups[i].append(task.id)
            load[i] += add
            end[i] = new_end
            speed = state.agents[i].speed or 1.0
            reasons[task.id] = f"свободная задача → аппарат {i}: закончит раньше всех (≈{finish / speed:.0f} с)"


# --------------------------------------------------------------------------- геометрия
def _polygon(task: "Task") -> Polygon:
    if len(task.polygon) >= 3:
        return Polygon(task.polygon)
    return Point(_centroid(task)).buffer(1.0)


def _centroid(task: "Task") -> tuple[float, float]:
    return tuple(task.centroid)


def _work_area(tasks: list["Task"]):
    area = unary_union([_polygon(t) for t in tasks]).buffer(0)
    return area if area.area > 0 else area.convex_hull.buffer(1.0)


def _match(sectors: list, positions: list[tuple[float, float]]) -> list[int]:
    """``owner[k]`` — индекс аппарата для сектора ``k``: минимум суммарного расстояния до центроидов."""
    n = len(sectors)

    def d(k: int, a: int) -> float:
        s = sectors[k]
        return 1e12 if s.is_empty else math.dist(positions[a], (s.centroid.x, s.centroid.y))

    if n <= _EXACT_MATCH_MAX:
        best = min(itertools.permutations(range(n)), key=lambda p: (sum(d(k, a) for k, a in enumerate(p)), p))
        return list(best)
    owner = [-1] * n
    taken: set[int] = set()
    pairs = sorted((d(k, a), k, a) for k in range(n) for a in range(n))
    for _, k, a in pairs:
        if owner[k] < 0 and a not in taken:
            owner[k] = a
            taken.add(a)
    return owner
