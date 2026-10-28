"""Адаптер: распределители М1 (``swarmsim.alloc``) → протокол симулятора (``swarmsim.interfaces``).

Контракты М1 и симулятора пока разные (это вопрос GEN-2, решается всей командой):

| | М1 (``alloc.py``) | симулятор (``interfaces.py``) |
|---|---|---|
| состояние | ``agent_ids``, ``domain_polygon()``, ``agent_position(id)`` | ``SwarmSnapshot`` (``agents[i].pos/alive``, ``world``) |
| id задачи | ``int`` | ``str`` (``"c_r00_c01"``) |
| результат | ``Assignment.by_agent`` — множество, порядок не маршрут | ``Assignment.plans`` — упорядоченная очередь |
| перераспределение | отдельный ``reassign(previous, failed_agents)`` | повторный ``allocate`` с текущими очередями в снимке |

Адаптер переводит одно в другое, ничего не меняя в коде М1: строковые id ↔ целые,
снимок → ``WorldStateLike``, множество задач → очередь через ``route_tasks`` (2-opt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from swarmsim import alloc as m1
from swarmsim.interfaces import Assignment, SwarmSnapshot
from swarmsim.plan import route_tasks

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.missions.base import Task, TaskSet


@dataclass(frozen=True)
class _TaskView:
    """Задача симулятора в виде ``TaskLike`` М1 (целый id, центроид, полигон)."""

    id: int
    centroid: tuple[float, float]
    polygon: Any = None


class _StateView:
    """``SwarmSnapshot`` в виде ``WorldStateLike`` М1: только живые агенты."""

    def __init__(self, snap: SwarmSnapshot) -> None:
        self._snap = snap
        self._domain = None

    @property
    def agent_ids(self) -> list[int]:
        return self._snap.alive_ids()

    def agent_position(self, agent_id: int) -> tuple[float, float]:
        return tuple(self._snap.agents[agent_id].pos)

    def domain_polygon(self):
        if self._domain is None:
            world = self._snap.world
            dom = Polygon(world.bounds)
            if world.nofly:
                dom = dom.difference(unary_union([Polygon(z) for z in world.nofly]))
            self._domain = dom
        return self._domain


class _BridgeBase:
    """Общая часть: перевод id, сохранение текущих очередей, порядок обхода."""

    name = "m1_bridge"
    requires_gcs = True
    #: умеет ли распределитель дораспределять задачи после отказа
    repairs = True

    def __init__(self, rng: Any = None, scenario: Any = None, **params: Any) -> None:
        # rng и scenario симулятор передаёт сам; алгоритмы М1 детерминированы и их не берут
        self.inner = self._make_inner(**params)

    def _make_inner(self, **params: Any):  # pragma: no cover - переопределяется
        raise NotImplementedError

    def should_trigger(self, t: float, tasks: "TaskSet", pending: list) -> bool:
        return bool(pending) or bool(tasks.unassigned())

    def allocate(self, state: SwarmSnapshot, tasks: "TaskSet") -> Assignment:
        alive = state.alive_ids()
        if not alive:
            return Assignment(plans={})

        unfinished = tasks.unfinished()
        str_of = {k: t.id for k, t in enumerate(unfinished)}
        int_of = {sid: k for k, sid in str_of.items()}
        views = [_TaskView(k, tuple(t.centroid), _polygon(t)) for k, t in enumerate(unfinished)]
        view_of = {v.id: v for v in views}
        sv = _StateView(state)

        # текущая задача и очередь живых агентов (то, что они уже делают)
        current: dict[int, str | None] = {}
        kept: dict[int, tuple[int, ...]] = {}
        for i in alive:
            v = state.agents[i]
            cur = v.current_task if v.current_task in int_of else None
            current[i] = cur
            kept[i] = tuple(int_of[tid] for tid in ([cur] if cur else []) + v.queue if tid in int_of)
        initial = not any(kept.values())

        reasons: dict[str, str] = {}
        if initial:
            asg = self.inner.allocate(sv, views)
            why = f"{self.name}: начальное распределение"
        elif self.repairs:
            asg = self.inner.reassign(sv, views, m1.Assignment(kept))
            why = f"{self.name}: дораспределение свободных задач"
        else:
            asg = m1.Assignment(kept)  # статический метод: без ремонта (так задумано для B1/B1b)
            why = ""

        plans: dict[int, list[str]] = {}
        for i in alive:
            ids = asg.tasks_of(i)
            cur = current.get(i)
            rest = [view_of[k] for k in ids if str_of[k] != cur]
            route = route_tasks(rest, start=tuple(state.agents[i].pos), agent_id=i)
            plans[i] = ([cur] if cur else []) + [str_of[k] for k in route.task_ids]
            for k in ids:
                if why and k not in kept.get(i, ()):
                    reasons[str_of[k]] = f"{why} → агент {i}"
        return Assignment(plans=plans, reasons=reasons, meta={"allocator": self.name})


def _polygon(task: "Task"):
    return Polygon(task.polygon) if len(task.polygon) >= 3 else None


class StaticNaiveSim(_BridgeBase):
    """B1 (полосы) в симуляторе. Без перераспределения: задачи отказавших остаются без исполнителя."""

    name = "static_naive"
    repairs = False

    def _make_inner(self, **params: Any):
        return m1.StaticNaiveAllocator(**params)


class StaticVoronoiSim(_BridgeBase):
    """B1b (Вороной + power diagram) в симуляторе. Без перераспределения."""

    name = "static_voronoi"
    repairs = False

    def _make_inner(self, **params: Any):
        return m1.StaticVoronoiAllocator(**params)


class CentralGreedySim(_BridgeBase):
    """B2 (центральный аукцион) в симуляторе: ``reassign`` после отказа или новой цели."""

    name = "central_greedy"

    def _make_inner(self, **params: Any):
        return m1.CentralGreedyAllocator(**params)
