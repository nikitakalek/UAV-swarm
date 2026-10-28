"""Порядок обхода задач одним аппаратом и оценка его загрузки (М1).

Задача миссии — «змейка» по ячейке: вход в первой точке (с учётом уже пройденной
части, ``Task.location``), выход в последней. Порядок строится ближайшим соседом и
улучшается 2-opt по полной стоимости маршрута «позиция → задачи → база».

Транзит между задачами оценивается по прямой. Обход бесполётных зон при исполнении
делает ``World.free_path``; его удлинение учитывает валидатор (``validator.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.missions.base import Task

Point = tuple[float, float]


@dataclass(frozen=True)
class RouteEstimate:
    """Оценка маршрута: порядок задач, длина (м) и время (с) до возврата на базу."""

    order: tuple[str, ...]
    length: float
    work: float  # длина рабочих галсов
    transit: float  # перелёты между задачами и домой

    def duration(self, speed: float) -> float:
        return self.length / speed


def _entry(task: "Task") -> Point:
    return tuple(task.location)


def _exit(task: "Task") -> Point:
    return tuple(task.waypoints[-1])


def route_cost(tasks: Sequence["Task"], start: Point, home: Optional[Point]) -> tuple[float, float]:
    """``(transit, work)`` для заданного порядка задач."""
    transit = work = 0.0
    pos = start
    for t in tasks:
        transit += math.dist(pos, _entry(t))
        work += t.remaining_length()
        pos = _exit(t)
    if home is not None:
        transit += math.dist(pos, home)
    return transit, work


def order_tasks(
    tasks: Sequence["Task"],
    start: Point,
    home: Optional[Point] = None,
    *,
    first: Optional[str] = None,
    max_passes: int = 20,
) -> RouteEstimate:
    """Порядок обхода: ближайший сосед + 2-opt. Детерминирован (при равенстве — меньший id).

    ``first`` — задача, которую аппарат уже выполняет: она остаётся первой.
    """
    pool = sorted(tasks, key=lambda t: t.id)
    head: list["Task"] = []
    if first is not None:
        head = [t for t in pool if t.id == first]
        pool = [t for t in pool if t.id != first]

    seq: list["Task"] = []
    pos = _exit(head[0]) if head else start
    while pool:
        k = min(range(len(pool)), key=lambda i: (math.dist(pos, _entry(pool[i])), pool[i].id))
        seq.append(pool.pop(k))
        pos = _exit(seq[-1])

    def cost(s: list["Task"]) -> float:
        return sum(route_cost(head + s, start, home))

    best = cost(seq)
    for _ in range(max_passes):
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = cost(cand)
                if c < best - 1e-9:
                    seq, best, improved = cand, c, True
        if not improved:
            break

    full = head + seq
    transit, work = route_cost(full, start, home)
    return RouteEstimate(tuple(t.id for t in full), transit + work, work, transit)


def append_cost(route_end: Point, task: "Task") -> tuple[float, Point]:
    """Сколько добавит задача, если поставить её в конец маршрута: ``(метры, новый конец)``."""
    return math.dist(route_end, _entry(task)) + task.remaining_length(), _exit(task)
