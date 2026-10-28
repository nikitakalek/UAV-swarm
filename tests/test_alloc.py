"""Тесты статических распределителей (M1-1 и M1-1a)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from shapely.geometry import box

from swarmsim.alloc import (
    CentralGreedyAllocator,
    OracleAllocator,
    StaticNaiveAllocator,
    StaticVoronoiAllocator,
)
from swarmsim.plan import route_assignment

AREA = 1000.0
N_VALUES = [5, 10, 20]


# --- заглушки (в реальной системе их даёт Модуль 2) ------------------------


@dataclass(frozen=True)
class StubTask:
    id: int
    centroid: tuple[float, float]


@dataclass(frozen=True)
class StubState:
    """Минимальный снимок мира: агенты, домен, стартовые позиции."""

    agent_ids: tuple[int, ...]
    _domain: object
    positions: dict = field(default_factory=dict)

    def domain_polygon(self):
        return self._domain

    def agent_position(self, agent_id: int):
        return self.positions[agent_id]


def make_tasks(nx: int = 25, ny: int = 25, size: float = AREA) -> list[StubTask]:
    """Сетка задач nx x ny — имитация выхода MissionPlugin (M2-5).

    Ячейки намеренно мельче самых узких секторов: в реальной системе
    MissionPlugin режет область по ширине полосы обзора, то есть ячейка
    сопоставима с сектором, а не крупнее его.
    """
    step = size / nx
    tasks: list[StubTask] = []
    tid = 0
    for iy in range(ny):
        for ix in range(nx):
            tasks.append(StubTask(tid, (step * (ix + 0.5), step * (iy + 0.5))))
            tid += 1
    return tasks


def make_state(n_agents: int, size: float = AREA) -> StubState:
    return StubState(tuple(range(n_agents)), box(0.0, 0.0, size, size))


def make_state_with_pads(n_agents: int, size: float = AREA) -> StubState:
    """Агенты на разных площадках вдоль нижней границы — нужно для M1-5."""
    span = size - 100.0
    denominator = max(1, n_agents - 1)
    positions = {
        i: (50.0 + span * i / denominator, 20.0) for i in range(n_agents)
    }
    return StubState(tuple(range(n_agents)), box(0.0, 0.0, size, size), positions)


def make_allocator(cls, **kwargs):
    if cls is StaticVoronoiAllocator:
        kwargs.setdefault("lloyd_iters", 30)
    return cls(**kwargs)


ALLOCATORS = [StaticNaiveAllocator, StaticVoronoiAllocator]


# --- корректность разбиения ------------------------------------------------


@pytest.mark.parametrize("n_agents", N_VALUES)
@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_every_task_assigned_exactly_once(cls, n_agents):
    tasks = make_tasks()
    state = make_state(n_agents)
    assignment = make_allocator(cls).allocate(state, tasks)

    assert set(assignment.agents()) == set(range(n_agents))
    assignment.check_partition([t.id for t in tasks])


@pytest.mark.parametrize("n_agents", N_VALUES)
@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_all_agents_get_work_when_tasks_are_enough(cls, n_agents):
    tasks = make_tasks()
    state = make_state(n_agents)
    assignment = make_allocator(cls).allocate(state, tasks)

    assert assignment.unassigned_agents() == (), "при 100 задачах работы должно хватить всем"


# --- детерминизм -----------------------------------------------------------


@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_determinism(cls):
    tasks = make_tasks()
    state = make_state(10)
    a1 = make_allocator(cls).allocate(state, tasks)
    a2 = make_allocator(cls).allocate(state, tasks)
    assert a1.by_agent == a2.by_agent


# --- крайние случаи --------------------------------------------------------


@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_more_agents_than_tasks(cls):
    """Агентов больше, чем задач: часть агентов без работы, но разбиение корректно."""
    tasks = [StubTask(i, (100.0 * i + 50.0, 50.0)) for i in range(4)]
    state = make_state(10)
    assignment = make_allocator(cls).allocate(state, tasks)

    assignment.check_partition([t.id for t in tasks])
    # 4 задачи могут занять не более 4 агентов — значит минимум 6 простаивают.
    # Точное число зависит от разбиения: у Вороного задачи могут лечь в 2 сектора.
    assert len(assignment.unassigned_agents()) >= 6


@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_single_agent_gets_all_tasks(cls):
    tasks = make_tasks()
    state = make_state(1)
    assignment = make_allocator(cls).allocate(state, tasks)

    assert len(assignment.tasks_of(0)) == len(tasks)
    assignment.check_partition([t.id for t in tasks])


@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_no_agents(cls):
    tasks = make_tasks()
    state = make_state(0)
    assignment = make_allocator(cls).allocate(state, tasks)
    assert assignment.by_agent == {}


@pytest.mark.parametrize("cls", ALLOCATORS, ids=lambda c: c.__name__)
def test_no_tasks(cls):
    state = make_state(5)
    assignment = make_allocator(cls).allocate(state, [])
    assert assignment.all_tasks() == ()
    assert len(assignment.unassigned_agents()) == 5


# --- квадрат области и «компактность» секторов -----------------------------


def _elongation(s) -> float:
    """Изопериметрическое отношение: 1.0 у круга, ~1.13 у квадрата, много больше у полосы."""
    return s.length / (4.0 * (s.area ** 0.5))


def _max_elongation(partition) -> float:
    return max(_elongation(s) for s in partition.sectors if not s.is_empty)


@pytest.mark.parametrize("n_agents", N_VALUES)
def test_voronoi_sectors_are_compact(n_agents):
    """Секторы Вороного должны быть близки к квадрату, а не к полосе."""
    alloc = make_allocator(StaticVoronoiAllocator)
    alloc.allocate(make_state(n_agents), make_tasks())
    q = _max_elongation(alloc.last_partition)
    assert q < 1.6, f"сектор слишком вытянут (q={q:.2f})"


@pytest.mark.parametrize("n_agents", [10, 20])
def test_voronoi_is_more_compact_than_naive(n_agents):
    """Главное утверждение M1-1: компактные секторы против вытянутых полос."""
    naive = StaticNaiveAllocator()
    naive.allocate(make_state(n_agents), make_tasks())
    vor = make_allocator(StaticVoronoiAllocator)
    vor.allocate(make_state(n_agents), make_tasks())

    q_naive = _max_elongation(naive.last_partition)
    q_vor = _max_elongation(vor.last_partition)
    print(f"\nn={n_agents}: вытянутость полос={q_naive:.2f}, секторов Вороного={q_vor:.2f}")
    assert q_naive > 1.5, "полосы должны быть вытянутыми"
    assert q_vor < q_naive, "секторы Вороного должны быть компактнее полос"


def test_area_stats_available_after_allocate():
    """Разброс площадей должен быть доступен для отчёта (критерий готовности M1-1)."""
    alloc = make_allocator(StaticVoronoiAllocator)
    alloc.allocate(make_state(10), make_tasks())
    stats = alloc.last_area_stats
    assert stats["n"] == 10
    assert stats["ratio"] >= 1.0
    assert stats["mean"] > 0


# --- центральный жадный аукцион (M1-5) -------------------------------------


@pytest.mark.parametrize("n_agents", N_VALUES)
def test_central_assigns_every_task_once(n_agents):
    tasks = make_tasks()
    assignment = CentralGreedyAllocator().allocate(make_state_with_pads(n_agents), tasks)
    assignment.check_partition([t.id for t in tasks])


@pytest.mark.parametrize("n_agents", N_VALUES)
def test_central_gives_work_to_all_agents(n_agents):
    tasks = make_tasks()
    assignment = CentralGreedyAllocator().allocate(make_state_with_pads(n_agents), tasks)
    assert assignment.unassigned_agents() == ()


def test_central_determinism():
    tasks = make_tasks()
    state = make_state_with_pads(10)
    first = CentralGreedyAllocator().allocate(state, tasks)
    second = CentralGreedyAllocator().allocate(state, tasks)
    assert first.by_agent == second.by_agent


def test_central_balances_load():
    """Загрузка не должна выродиться в «всё одному агенту»."""
    tasks = make_tasks()
    assignment = CentralGreedyAllocator().allocate(make_state_with_pads(10), tasks)
    sizes = sorted(assignment.sizes().values())
    mean = sum(sizes) / len(sizes)
    assert sizes[0] > 0
    assert sizes[-1] < 3.0 * mean, f"перекос загрузки: {sizes}"


def test_central_requires_positions():
    """Без стартовых позиций жадная вставка вырождается — требуем их явно."""

    class NoPositions:
        agent_ids = (0, 1)

        def domain_polygon(self):
            return box(0.0, 0.0, AREA, AREA)

    with pytest.raises(AttributeError):
        CentralGreedyAllocator().allocate(NoPositions(), make_tasks())


def test_central_no_agents():
    state = StubState((), box(0.0, 0.0, AREA, AREA), {})
    assert CentralGreedyAllocator().allocate(state, make_tasks()).by_agent == {}


def test_central_no_tasks():
    assignment = CentralGreedyAllocator().allocate(make_state_with_pads(5), [])
    assert assignment.all_tasks() == ()
    assert len(assignment.unassigned_agents()) == 5


# --- пересчёт назначения (события) -----------------------------------------


def test_reassign_after_failure_redistributes():
    tasks = make_tasks()
    state = make_state_with_pads(10)
    alloc = CentralGreedyAllocator()
    before = alloc.allocate(state, tasks)

    after = alloc.reassign(state, tasks, before, failed_agents=(3,))

    assert 3 not in after.by_agent
    after.check_partition([t.id for t in tasks])
    assert after.unassigned_agents() == ()


def test_reassign_keeps_healthy_agents():
    tasks = make_tasks()
    state = make_state_with_pads(10)
    alloc = CentralGreedyAllocator()
    before = alloc.allocate(state, tasks)

    after = alloc.reassign(state, tasks, before, failed_agents=(2,))

    for agent in before.by_agent:
        if agent == 2:
            continue
        assert set(before.tasks_of(agent)) <= set(after.tasks_of(agent))


def test_reassign_adds_new_tasks():
    tasks = make_tasks()
    base = tasks[:600]  # последняя строка из 25 задач — «новые цели»
    state = make_state_with_pads(10)
    alloc = CentralGreedyAllocator()
    before = alloc.allocate(state, base)

    after = alloc.reassign(state, tasks, before)

    after.check_partition([t.id for t in tasks])
    assert len(after.all_tasks()) == 625


# --- потолок качества (M1-2) -----------------------------------------------


def small_tasks():
    return make_tasks(nx=5, ny=4)


def makespan_of(assignment, tasks, state):
    by_id = {t.id: t for t in tasks}
    positions = {a: state.agent_position(a) for a in state.agent_ids}
    routes = route_assignment(assignment, by_id, positions, homes=positions)
    return max(r.length for r in routes.values())


def test_oracle_partition():
    tasks = small_tasks()
    state = make_state_with_pads(4)
    assignment = OracleAllocator(max_passes=2).allocate(state, tasks)
    assignment.check_partition([t.id for t in tasks])


def test_oracle_not_worse_than_naive():
    """Главное требование к оракулу: он не хуже наивного метода."""
    tasks = small_tasks()
    state = make_state_with_pads(4)
    naive = StaticNaiveAllocator().allocate(state, tasks)
    oracle = OracleAllocator(max_passes=2).allocate(state, tasks)
    assert makespan_of(oracle, tasks, state) <= makespan_of(naive, tasks, state) + 1e-6


def test_oracle_not_worse_than_central():
    tasks = small_tasks()
    state = make_state_with_pads(4)
    central = CentralGreedyAllocator().allocate(state, tasks)
    oracle = OracleAllocator(max_passes=2).allocate(state, tasks)
    assert makespan_of(oracle, tasks, state) <= makespan_of(central, tasks, state) + 1e-6


def test_oracle_determinism():
    tasks = small_tasks()
    state = make_state_with_pads(4)
    first = OracleAllocator(max_passes=2).allocate(state, tasks)
    second = OracleAllocator(max_passes=2).allocate(state, tasks)
    assert first.by_agent == second.by_agent


def test_oracle_no_tasks():
    assignment = OracleAllocator().allocate(make_state_with_pads(3), [])
    assert assignment.all_tasks() == ()


def test_oracle_requires_positions():
    class NoPositions:
        agent_ids = (0, 1)

        def domain_polygon(self):
            return box(0.0, 0.0, AREA, AREA)

    with pytest.raises(AttributeError):
        OracleAllocator().allocate(NoPositions(), small_tasks())
