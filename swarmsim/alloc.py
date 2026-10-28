"""Распределение задач между агентами (Модуль 1).

Здесь контракты (`Assignment`, `Allocator`, `WorldStateLike`, `TaskLike`) и
статические распределители:

* :class:`StaticNaiveAllocator` — полосы равной площади (baseline B1, M1-1a);
* :class:`StaticVoronoiAllocator` — Вороной + power diagram (B1b, M1-1).

Контракты — «стык» между модулями (см. docs/02_mvp_architecture.md, P0-2).
Их изменение возможно только решением всей команды.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from shapely.geometry import Polygon

from .sectors import (
    SectorPartition,
    area_stats,
    assign_tasks,
    build_partition,
    strip_partition,
)

AgentId = int
TaskId = int


# ---------------------------------------------------------------------------
# Контракты
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assignment:
    """Результат распределения: какие задачи достались какому агенту.

    Порядок задач внутри агента **не** является маршрутом — маршрутизацию
    (порядок обхода с учётом энергии) выполняет отдельный модуль M1-3.
    Здесь задачи хранятся отсортированными для детерминизма.
    """

    by_agent: Mapping[AgentId, tuple[TaskId, ...]]

    def agents(self) -> tuple[AgentId, ...]:
        return tuple(self.by_agent.keys())

    def tasks_of(self, agent: AgentId) -> tuple[TaskId, ...]:
        return tuple(self.by_agent.get(agent, ()))

    def all_tasks(self) -> tuple[TaskId, ...]:
        out: list[TaskId] = []
        for v in self.by_agent.values():
            out.extend(v)
        return tuple(out)

    def sizes(self) -> dict[AgentId, int]:
        return {a: len(v) for a, v in self.by_agent.items()}

    def unassigned_agents(self) -> tuple[AgentId, ...]:
        return tuple(a for a, v in self.by_agent.items() if not v)

    def check_partition(self, expected_task_ids: Sequence[TaskId]) -> None:
        """Проверяет корректность разбиения: полнота и отсутствие дублей.

        Бросает ``AssertionError`` с перечислением проблем.
        """
        got = list(self.all_tasks())
        counts: dict[TaskId, int] = {}
        for t in got:
            counts[t] = counts.get(t, 0) + 1

        dupes = sorted(t for t, c in counts.items() if c > 1)
        expected = set(expected_task_ids)
        missing = sorted(expected - set(got))
        extra = sorted(set(got) - expected)

        problems: list[str] = []
        if dupes:
            problems.append(f"задачи назначены нескольким агентам: {dupes}")
        if missing:
            problems.append(f"не назначены: {missing}")
        if extra:
            problems.append(f"лишние (нет в наборе задач): {extra}")
        if problems:
            raise AssertionError("Нарушено разбиение задач: " + "; ".join(problems))


@runtime_checkable
class WorldStateLike(Protocol):
    """Минимальный «снимок мира», который нужен статическому распределителю.

    Требование к Модулю 2 (M2-2): ``World`` должен отдавать идентификаторы
    агентов и геометрию домена (область минус бесполётные зоны).
    """

    @property
    def agent_ids(self) -> Sequence[AgentId]: ...

    def domain_polygon(self):  # -> shapely Polygon/MultiPolygon
        ...


@runtime_checkable
class TaskLike(Protocol):
    """Минимальное требование к задаче (создаётся MissionPlugin, M2-5)."""

    @property
    def id(self) -> TaskId: ...

    @property
    def centroid(self) -> tuple[float, float]: ...


@runtime_checkable
class Allocator(Protocol):
    """Единый интерфейс всех распределителей (B0..B5)."""

    name: str

    def allocate(self, state: WorldStateLike, tasks: Sequence[TaskLike]) -> Assignment: ...


# ---------------------------------------------------------------------------
# Статические распределители
# ---------------------------------------------------------------------------


def _pad(partition: SectorPartition, n: int) -> SectorPartition:
    """Дополняет разбиение пустыми секторами, если секторов получилось меньше N.

    Это возможно на невыпуклых доменах и с бесполётными зонами.
    """
    sectors = list(partition.sectors)
    seeds = list(partition.seeds)
    while len(sectors) < n:
        sectors.append(Polygon())
        seeds.append((float("nan"), float("nan")))
    return SectorPartition(tuple(sectors[:n]), tuple(seeds[:n]))


@dataclass
class StaticNaiveAllocator:
    """B1: полосы равной площади, один агент — одна полоса (M1-1a).

    Метод намеренно наивный: без выравнивания площадей, без учёта энергии,
    без перераспределения. Нужен как нижняя граница и кадр «до» в демонстрации.
    """

    axis: str = "long"
    assign: str = "centroid"
    name: str = "static_naive"

    # диагностика последнего вызова (для отчёта и визуализации)
    last_partition: SectorPartition | None = field(default=None, repr=False)
    last_area_stats: dict = field(default_factory=dict, repr=False)

    def allocate(self, state: WorldStateLike, tasks: Sequence[TaskLike]) -> Assignment:
        agent_ids = list(state.agent_ids)
        if not agent_ids:
            return Assignment({})

        domain = state.domain_polygon()
        partition = strip_partition(domain, len(agent_ids), axis=self.axis)
        partition = _pad(partition, len(agent_ids))

        self.last_partition = partition
        self.last_area_stats = area_stats(partition)

        groups = assign_tasks(tasks, partition.sectors, mode=self.assign)
        by_agent: dict[AgentId, tuple[int, ...]] = {
            aid: tuple(groups[i]) for i, aid in enumerate(agent_ids)
        }
        return Assignment(by_agent)


@dataclass
class StaticVoronoiAllocator:
    """B1b: Вороной + power diagram — компактные секторы равной площади (M1-1).

    Конвейер: решётка семян → Ллойд (компактность) → power diagram (точные
    площади). ``power_iters=0`` отключает выравнивание площадей.
    """

    lattice: str = "grid"  # "grid" (квадратная) или "hex" (соты)
    lloyd_iters: int = 50
    power_iters: int = 60
    assign: str = "centroid"
    name: str = "static_voronoi"

    last_partition: SectorPartition | None = field(default=None, repr=False)
    last_area_stats: dict = field(default_factory=dict, repr=False)

    def allocate(self, state: WorldStateLike, tasks: Sequence[TaskLike]) -> Assignment:
        agent_ids = list(state.agent_ids)
        if not agent_ids:
            return Assignment({})

        domain = state.domain_polygon()
        partition = build_partition(
            domain,
            len(agent_ids),
            lattice=self.lattice,
            lloyd_iters=self.lloyd_iters,
            power_iters=self.power_iters,
        )
        partition = _pad(partition, len(agent_ids))

        self.last_partition = partition
        self.last_area_stats = area_stats(partition)

        groups = assign_tasks(tasks, partition.sectors, mode=self.assign)
        by_agent: dict[AgentId, tuple[int, ...]] = {
            aid: tuple(groups[i]) for i, aid in enumerate(agent_ids)
        }
        return Assignment(by_agent)


# ---------------------------------------------------------------------------
# Центральный жадный аукцион (B2, M1-5)
# ---------------------------------------------------------------------------


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(_dist(a, b) for a, b in zip(points, points[1:]))


@dataclass
class CentralGreedyAllocator:
    """B2: центральный жадный аукцион с пулом задач (M1-5).

    Аукцион идёт раундами: в каждом раунде агенты по очереди забирают из пула
    **ближайшую к концу своего маршрута** задачу. Это даёт сбалансированную
    загрузку (разница не больше одной задачи), локальные маршруты и детерминизм.

    Наивная «глобальная cheapest insertion» здесь не годится: первый агент
    удлиняет маршрут дешевле, чем остальные стартуют со своих площадок, и
    забирает почти все задачи (на проверке — 602 из 625).

    ``reassign`` возвращает в пул задачи отказавших агентов и «новые цели»,
    после чего дораспределяет пул тем же правилом.

    Энергия здесь не ограничивается: недопустимые маршруты ловит `PlanValidator`.
    """

    name: str = "central_greedy"

    # диагностика последнего вызова
    last_cost: float = field(default=0.0, init=False, repr=False)
    last_counts: dict[AgentId, int] = field(default_factory=dict, init=False, repr=False)

    def allocate(self, state: WorldStateLike, tasks: Sequence[TaskLike]) -> Assignment:
        agent_ids = list(state.agent_ids)
        if not agent_ids:
            return Assignment({})

        domain = state.domain_polygon()
        routes = {a: [self._start(state, a, domain)] for a in agent_ids}
        by_agent: dict[AgentId, list[int]] = {a: [] for a in agent_ids}
        remaining = list(tasks)

        # раунды: в каждом агенты по очереди берут ближайшую к концу маршрута задачу
        while remaining:
            for agent in agent_ids:
                if not remaining:
                    break
                index = self._take_nearest(routes[agent][-1], remaining)
                task = remaining.pop(index)
                routes[agent].append(tuple(task.centroid))
                by_agent[agent].append(int(task.id))

        self.last_cost = sum(_polyline_length(r) for r in routes.values())
        self.last_counts = {a: len(v) for a, v in by_agent.items()}
        return Assignment({a: tuple(sorted(v)) for a, v in by_agent.items()})

    def reassign(
        self,
        state: WorldStateLike,
        tasks: Sequence[TaskLike],
        previous: Assignment,
        *,
        failed_agents: Sequence[AgentId] = (),
    ) -> Assignment:
        """Пересчитывает назначение после событий.

        Задачи отказавших агентов возвращаются в пул; задачи, которых нет в
        ``previous`` (новые цели), тоже попадают в пул. Здоровые агенты сохраняют
        свои назначения — пересчитываются только «сироты».
        """
        failed = set(failed_agents)
        agent_ids = [a for a in state.agent_ids if a not in failed]
        if not agent_ids:
            return Assignment({})

        domain = state.domain_polygon()
        by_id = {int(t.id): t for t in tasks}

        kept: dict[AgentId, list[int]] = {}
        assigned: set[int] = set()
        for agent in agent_ids:
            ids = [t for t in previous.tasks_of(agent) if t in by_id]
            kept[agent] = ids
            assigned.update(ids)

        pool = [t for t in tasks if int(t.id) not in assigned]

        # текущие маршруты здоровых — в детерминированном пространственном порядке
        routes = {}
        for agent in agent_ids:
            points = sorted(
                (tuple(by_id[t].centroid) for t in kept[agent]),
                key=lambda p: (round(p[1], 9), p[0]),
            )
            routes[agent] = [self._start(state, agent, domain)] + points

        while pool:
            for agent in agent_ids:
                if not pool:
                    break
                index = self._take_nearest(routes[agent][-1], pool)
                task = pool.pop(index)
                routes[agent].append(tuple(task.centroid))
                kept[agent].append(int(task.id))

        self.last_counts = {a: len(v) for a, v in kept.items()}
        return Assignment({a: tuple(sorted(v)) for a, v in kept.items()})

    # -- внутреннее ---------------------------------------------------------

    def _start(self, state: WorldStateLike, agent_id: AgentId, domain):
        getter = getattr(state, "agent_position", None)
        if not callable(getter):
            raise AttributeError(
                "CentralGreedyAllocator требует state.agent_position(agent_id): "
                "без стартовых позиций агентов аукцион вырождается "
                "(все задачи достаются одному агенту)"
            )
        return tuple(getter(agent_id))

    @staticmethod
    def _take_nearest(
        route_end: tuple[float, float],
        pool: Sequence[TaskLike],
    ) -> int:
        """Индекс ближайшей к ``route_end`` задачи; при равенстве — меньший id."""
        best_index = 0
        best_key: tuple[float, int] | None = None
        for i, task in enumerate(pool):
            key = (_dist(route_end, tuple(task.centroid)), int(task.id))
            if best_key is None or key < best_key:
                best_key, best_index = key, i
        return best_index


# ---------------------------------------------------------------------------
# Офлайн-потолок качества (B0, M1-2)
# ---------------------------------------------------------------------------


@dataclass
class OracleAllocator:
    """B0: потолок качества — эвристический офлайн-оптимум (M1-2).

    Стартует с **лучшего** из двух кандидатов — полос (B1) и центрального
    аукциона (B2) — и улучшает назначение переносом задач между агентами,
    минимизируя максимальную длину замкнутого маршрута (makespan). Поэтому он
    гарантированно не хуже B1 и B2, что и требуется от потолка.

    Это **эвристический** потолок, а не сертифицированный оптимум: точный mTSP
    для M = 40 за разумное время не решается. OR-Tools дал бы более сильный
    результат, но требует тяжёлой зависимости — оставлено на будущее
    (см. docs/04_algorithms.md). Предназначен для M ≤ 40.

    Позиции агентов обязательны. Если у ``state`` есть ``agent_energy``, то
    энергия учитывается: недопустимые по бюджету назначения отбрасываются.
    """

    max_passes: int = 5
    minimize: str = "makespan"  # "makespan" | "total"
    reserve: float = 0.2
    name: str = "static_oracle"

    last_makespan: float = field(default=0.0, init=False, repr=False)
    last_total: float = field(default=0.0, init=False, repr=False)

    def allocate(self, state: WorldStateLike, tasks: Sequence[TaskLike]) -> Assignment:
        from .plan import route_tasks  # ленивый импорт: модули не циклируются

        agent_ids = list(state.agent_ids)
        if not agent_ids:
            return Assignment({})

        getter = getattr(state, "agent_position", None)
        if not callable(getter):
            raise AttributeError("OracleAllocator требует state.agent_position(agent_id)")
        starts = {a: tuple(getter(a)) for a in agent_ids}

        energy_getter = getattr(state, "agent_energy", None)
        budgets = {
            a: (energy_getter(a) if callable(energy_getter) else None) for a in agent_ids
        }

        tasks = list(tasks)
        by_id = {int(t.id): t for t in tasks}

        def evaluate(groups: dict[AgentId, list[int]]):
            lengths: list[float] = []
            total = 0.0
            feasible = True
            for agent in agent_ids:
                route = route_tasks(
                    [by_id[t] for t in groups[agent]],
                    starts[agent],
                    home=starts[agent],
                    budget=budgets[agent],
                    reserve=self.reserve,
                )
                lengths.append(route.length)
                total += route.length
                feasible = feasible and route.feasible
            return max(lengths), total, feasible

        def score(makespan: float, total: float, feasible: bool) -> tuple:
            """Лексикографический критерий: сначала допустимость, потом главная метрика."""
            if not feasible:
                return (1, 0.0, 0.0)
            if self.minimize == "total":
                return (0, total, makespan)
            return (0, makespan, total)

        # кандидаты: полосы (B1) и центральный аукцион (B2)
        candidates = (
            StaticNaiveAllocator().allocate(state, tasks),
            CentralGreedyAllocator().allocate(state, tasks),
        )

        best_groups: dict[AgentId, list[int]] = {a: [] for a in agent_ids}
        best_value = (1, 0.0, 0.0)
        for candidate in candidates:
            groups = {a: list(candidate.tasks_of(a)) for a in agent_ids}
            value = score(*evaluate(groups))
            if value < best_value:
                best_groups, best_value = groups, value

        # локальный поиск: перенос задачи к другому агенту и обмен задачами
        groups = {a: list(v) for a, v in best_groups.items()}
        for _ in range(max(0, self.max_passes)):
            improved = False

            # 1) перенос: задача уходит к другому агенту
            for source in agent_ids:
                for task_id in list(groups[source]):
                    for target in agent_ids:
                        if target == source:
                            continue
                        groups[source].remove(task_id)
                        groups[target].append(task_id)
                        value = score(*evaluate(groups))
                        if value < best_value:
                            best_value = value
                            improved = True
                            break
                        groups[target].remove(task_id)
                        groups[source].append(task_id)

            # 2) обмен: две задачи меняются агентами
            for i, left in enumerate(agent_ids):
                for right in agent_ids[i + 1 :]:
                    done = False
                    for left_task in list(groups[left]):
                        if done:
                            break
                        for right_task in list(groups[right]):
                            groups[left].remove(left_task)
                            groups[right].remove(right_task)
                            groups[left].append(right_task)
                            groups[right].append(left_task)
                            value = score(*evaluate(groups))
                            if value < best_value:
                                best_value = value
                                improved = True
                                done = True
                                break
                            groups[left].remove(right_task)
                            groups[right].remove(left_task)
                            groups[left].append(left_task)
                            groups[right].append(right_task)

            if not improved:
                break

        makespan, total, _ = evaluate(groups)
        self.last_makespan = makespan
        self.last_total = total
        return Assignment({a: tuple(sorted(groups[a])) for a in agent_ids})
