"""API плагина миссии и набор задач ``TaskSet`` (M2-5).

Миссия — доменный плагин: она превращает описание миссии в набор задач, а ядро
дальше работает только с задачами (docs/02, антипаттерн 7).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator, Optional

from swarmsim import geometry as geo

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.scenario import MissionConfig
    from swarmsim.world import World


class TaskStatus(str, Enum):
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    kind: str
    waypoints: list[tuple[float, float]]
    polygon: list[tuple[float, float]] = field(default_factory=list)
    priority: float = 1.0
    deadline: Optional[float] = None
    sector: Optional[str] = None
    status: TaskStatus = TaskStatus.UNASSIGNED
    assignee: Optional[int] = None
    progress: int = 0  # индекс следующей непройденной точки
    t_created: float = 0.0
    t_assigned: Optional[float] = None
    t_started: Optional[float] = None
    t_done: Optional[float] = None
    done_count: int = 0
    done_by: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED)

    @property
    def location(self) -> tuple[float, float]:
        """Точка входа в задачу (используется распределителями для оценки пути)."""
        return self.waypoints[min(self.progress, len(self.waypoints) - 1)]

    @property
    def centroid(self) -> tuple[float, float]:
        if len(self.polygon) >= 3:
            return geo.centroid(self.polygon)
        xs, ys = zip(*self.waypoints)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def length(self) -> float:
        return geo.path_length(self.waypoints)

    def remaining_length(self) -> float:
        return geo.path_length(self.waypoints[self.progress:]) if self.progress < len(self.waypoints) else 0.0

    def mark_assigned(self, agent_id: Optional[int], t: float) -> None:
        if self.is_finished:
            return
        self.assignee = agent_id
        if agent_id is None:
            self.status = TaskStatus.UNASSIGNED
        else:
            self.status = TaskStatus.ASSIGNED
            self.t_assigned = t

    def mark_started(self, agent_id: int, t: float) -> None:
        self.assignee = agent_id
        self.status = TaskStatus.IN_PROGRESS
        if self.t_started is None:
            self.t_started = t

    def mark_done(self, agent_id: int, t: float) -> None:
        self.done_count += 1
        if self.status == TaskStatus.DONE:
            return
        self.status = TaskStatus.DONE
        self.progress = len(self.waypoints)
        self.t_done = t
        self.done_by = agent_id

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "status": self.status.value, "assignee": self.assignee,
            "done_by": self.done_by, "priority": self.priority, "sector": self.sector,
            "t_created": self.t_created, "t_assigned": self.t_assigned, "t_started": self.t_started,
            "t_done": self.t_done, "done_count": self.done_count, "length_m": round(self.length, 3),
            "polygon": [list(map(lambda v: round(v, 3), p)) for p in self.polygon],
            "waypoints": [list(map(lambda v: round(v, 3), p)) for p in self.waypoints],
        }


class TaskSet:
    """Упорядоченный по id набор задач со статусами."""

    def __init__(self, tasks: Optional[list[Task]] = None) -> None:
        self._tasks: dict[str, Task] = {}
        for t in tasks or []:
            self.add(t)

    def add(self, task: Task) -> Task:
        if task.id in self._tasks:
            raise ValueError(f"дубликат id задачи: {task.id}")
        self._tasks[task.id] = task
        return task

    def __getitem__(self, tid: str) -> Task:
        return self._tasks[tid]

    def __contains__(self, tid: str) -> bool:
        return tid in self._tasks

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def tasks(self) -> dict[str, Task]:
        return self._tasks

    def ids(self) -> list[str]:
        return list(self._tasks)

    def unfinished(self) -> list[Task]:
        return [t for t in self if not t.is_finished]

    def unassigned(self) -> list[Task]:
        return [t for t in self if t.status == TaskStatus.UNASSIGNED]

    def by_status(self, status: TaskStatus) -> list[Task]:
        return [t for t in self if t.status == status]

    def completion_rate(self) -> float:
        return sum(t.status == TaskStatus.DONE for t in self) / len(self) if len(self) else math.nan


class MissionPlugin(ABC):
    """Плагин миссии: ``decompose(mission, world) -> TaskSet``."""

    name: str = "base"

    @abstractmethod
    def decompose(self, mission: "MissionConfig", world: "World") -> TaskSet: ...

    def on_new_target(self, mission: "MissionConfig", world: "World", tasks: TaskSet,
                      pos: tuple[float, float], radius: float, t: float) -> list[Task]:
        """Дополнительные задачи по событию «новая цель» (по умолчанию — нет)."""
        return []
