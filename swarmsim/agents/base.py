"""Абстракция агента и источника движения (M2-3, M2-11).

``MotionSource`` — единственная граница между логикой роя и «железом». За ней
стоит либо ``KinematicMotion`` (симуляция), либо ``swarmcore_px4.MavsdkMotion``
(PX4 SITL / борт). ``Agent`` ничего не знает о том, какая реализация внутри.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from swarmsim.interfaces import State

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.missions.base import TaskSet
    from swarmsim.world import World


class MotionSource(ABC):
    """Источник движения одного аппарата.

    Маршрут — список точек ``(x, y)`` в локальной системе (x — восток, y — север, м).
    ``step(dt)`` продвигает аппарат на шаг и возвращает ``State``; поле
    ``State.route_index`` — число уже достигнутых точек текущего маршрута.
    """

    agent_id: int

    @abstractmethod
    def state(self) -> State: ...

    @abstractmethod
    def step(self, dt: float) -> State: ...

    @abstractmethod
    def set_route(self, points: Sequence[tuple[float, float]]) -> None: ...

    def goto(self, x: float, y: float) -> None:
        self.set_route([(x, y)])

    @abstractmethod
    def arm(self) -> None: ...

    @abstractmethod
    def takeoff(self, alt: float) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def rtl(self) -> None: ...

    @abstractmethod
    def hold(self) -> None: ...

    def kill(self) -> None:
        """Имитация отказа: аппарат перестаёт двигаться и отвечать."""
        self.hold()

    def close(self) -> None:
        pass


@dataclass
class RoutePoint:
    x: float
    y: float
    task_id: Optional[str] = None
    wp_index: Optional[int] = None  # индекс точки задачи; None — транзит


@dataclass
class Agent:
    """Локальный цикл агента: следование плану (очереди задач) и возврат домой."""

    id: int
    motion: MotionSource
    home: tuple[float, float]
    alive: bool = True
    queue: list[str] = field(default_factory=list)
    current: Optional[str] = None
    energy_used_wh: float = 0.0
    distance_m: float = 0.0
    status: str = "idle"  # idle | transit | working | returning | home | failed
    _route: list[RoutePoint] = field(default_factory=list)
    _route_done: int = 0
    prev_state: Optional[State] = None
    state: Optional[State] = None

    def __post_init__(self) -> None:
        self.state = self.motion.state()
        self.prev_state = self.state

    # ------------------------------------------------------------------ план
    def set_plan(self, task_ids: list[str], tasks: "TaskSet") -> None:
        """Принять новый упорядоченный план. Текущая задача продолжается, если осталась в плане."""
        plan = [t for t in task_ids if not tasks[t].is_finished]
        if self.current is not None and self.current in plan:
            plan.remove(self.current)
            self.queue = plan
            return
        if self.current is not None:
            self._abandon_current(tasks)
        self.queue = plan

    def release_all(self, tasks: "TaskSet") -> list[str]:
        released = []
        if self.current is not None:
            released.append(self.current)
            self._abandon_current(tasks)
        released.extend(self.queue)
        self.queue = []
        return released

    def _abandon_current(self, tasks: "TaskSet") -> None:
        self.current = None
        self._route = []
        self._route_done = 0

    def fail(self) -> None:
        self.alive = False
        self.status = "failed"
        self.motion.kill()

    # ------------------------------------------------------------------ шаг
    def step(self, dt: float, tasks: "TaskSet", world: "World") -> State:
        self.prev_state = self.state
        if not self.alive:
            self.state = self.motion.step(dt)
            self.state.mode = "failed"
            return self.state
        if self.current is None:
            self._start_next(tasks, world)
        st = self.motion.step(dt)
        self._advance(st.route_index, st.t, tasks)
        if self.current is None and not self.queue and self.status in ("working", "transit"):
            self._go_home(world)
        if self.status == "returning" and st.route_index >= len(self._route):
            self.status = "home"
        if self.prev_state is not None:
            self.distance_m += math.dist(self.prev_state.pos, st.pos)
        self.state = st
        return st

    def _start_next(self, tasks: "TaskSet", world: "World") -> None:
        while self.queue:
            tid = self.queue.pop(0)
            task = tasks[tid]
            if task.is_finished:
                continue
            self.current = tid
            task.mark_started(self.id, self.state.t if self.state else 0.0)
            pos = self.state.pos
            remaining = task.waypoints[task.progress:]
            route: list[RoutePoint] = []
            for p in world.free_path(pos, remaining[0])[:-1]:
                route.append(RoutePoint(p[0], p[1]))
            prev = remaining[0]
            for k, wp in enumerate(remaining):
                idx = task.progress + k
                if k > 0:
                    for p in world.free_path(prev, wp)[:-1]:
                        route.append(RoutePoint(p[0], p[1], tid, None))
                route.append(RoutePoint(wp[0], wp[1], tid, idx))
                prev = wp
            self._set_route(route)
            self.status = "transit"
            return
        if self.status not in ("home", "returning", "idle"):
            self._go_home(world)

    def _go_home(self, world: "World") -> None:
        pts = world.free_path(self.state.pos, self.home)
        self._set_route([RoutePoint(p[0], p[1]) for p in pts])
        self.status = "returning"

    def _set_route(self, route: list[RoutePoint]) -> None:
        self._route = route
        self._route_done = 0
        self.motion.set_route([(r.x, r.y) for r in route])

    def _advance(self, route_index: int, t: float, tasks: "TaskSet") -> None:
        while self._route_done < min(route_index, len(self._route)):
            rp = self._route[self._route_done]
            self._route_done += 1
            if rp.task_id is None or rp.task_id != self.current:
                continue
            task = tasks[rp.task_id]
            if rp.wp_index is not None:
                self.status = "working"
                task.progress = max(task.progress, rp.wp_index + 1)
                if task.progress >= len(task.waypoints):
                    task.mark_done(self.id, t)
                    self.current = None
        if self.current is not None and self._route_done >= len(self._route):
            # маршрут исчерпан, а задача не закрыта (например, точка недостижима)
            tasks[self.current].mark_done(self.id, t)
            self.current = None

    # ------------------------------------------------------------------ сводка
    def remaining_work_m(self, tasks: "TaskSet") -> float:
        total = 0.0
        for tid in ([self.current] if self.current else []) + self.queue:
            total += tasks[tid].remaining_length()
        return total
