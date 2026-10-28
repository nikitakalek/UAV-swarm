"""Стыки между модулями (GEN-2). Меняются только решением всей команды.

Здесь описаны *контракты*, а не реализации:

* ``Allocator.allocate(state, tasks) -> Assignment``   — М1 (алгоритмы);
* ``MotionSource.step(dt) -> State``                    — М2 (``swarmsim.agents``);
* ``CommsBus.send / broadcast``                         — М3 (связь);
* ``World.observe(agent, sensor) -> Detection``         — М2 (среда) + М3 (модель Pd);
* ``FaultManager.inject / detect``                      — М3 (отказы);
* ``EnergyModel.power(v)``                              — М3 (энергия);
* ``MetricsRecorder.record(...) -> rows``               — интеграция (INT-1).

Симулятор (``swarmsim.sim``) зависит только от этих протоколов, поэтому модули
М1/М3 подключаются без правки ядра.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.agents.base import Agent
    from swarmsim.eventlog import Event
    from swarmsim.missions.base import TaskSet
    from swarmsim.world import Target, World


# --------------------------------------------------------------------------- движение
@dataclass
class State:
    """Состояние агента, которое отдаёт источник движения."""

    agent_id: int
    t: float
    x: float
    y: float
    alt: float
    vx: float = 0.0
    vy: float = 0.0
    armed: bool = True
    mode: str = "idle"  # idle | flying | landed | failed
    route_index: int = 0  # сколько точек текущего маршрута уже пройдено

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


# --------------------------------------------------------------------------- распределение
@dataclass
class AgentView:
    """Что распределитель знает об агенте в момент вызова."""

    agent_id: int
    pos: tuple[float, float]
    home: tuple[float, float]
    alive: bool
    speed: float
    current_task: Optional[str]
    queue: list[str]
    energy_used_wh: float


@dataclass
class SwarmSnapshot:
    """Аргумент ``state`` для ``Allocator.allocate``."""

    t: float
    agents: dict[int, AgentView]
    world: "World"
    gcs_reachable: bool = True

    def alive_ids(self) -> list[int]:
        return sorted(i for i, a in self.agents.items() if a.alive)


@dataclass
class Assignment:
    """Результат распределения: упорядоченный список задач на каждого агента.

    ``reasons`` — необязательное объяснение «почему» по задаче (попадает в журнал
    объяснений, M2-8).
    """

    plans: dict[int, list[str]] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def owner_of(self) -> dict[str, int]:
        return {tid: aid for aid, tids in self.plans.items() for tid in tids}


@runtime_checkable
class Allocator(Protocol):
    name: str

    def allocate(self, state: SwarmSnapshot, tasks: "TaskSet") -> Assignment: ...


# Необязательные методы распределителя (проверяются через getattr):
#   should_trigger(t, tasks, pending_events) -> bool   — по модельному времени/событиям
#   requires_gcs: bool = True                          — нужен ли канал с наземной станцией
#   on_message(msg) -> None                            — для децентрализованных схем


# --------------------------------------------------------------------------- сенсор / обнаружение
@dataclass
class Hit:
    target_id: int
    p: float  # вероятность обнаружения на этом шаге
    detected: bool  # исход розыгрыша


@dataclass
class Detection:
    """Результат ``World.observe(agent, sensor)`` за один шаг."""

    t: float
    agent_id: int
    hits: list[Hit] = field(default_factory=list)

    @property
    def new_detections(self) -> list[int]:
        return [h.target_id for h in self.hits if h.detected]


@runtime_checkable
class Sensor(Protocol):
    swath_width: float

    def detect_prob(self, agent: "Agent", target: "Target", world: "World") -> float: ...


# --------------------------------------------------------------------------- связь
GCS_ID = -1  # адрес наземной станции


@dataclass
class Message:
    src: int
    dst: Optional[int]  # None — широковещательное
    kind: str
    payload: dict[str, Any]
    size: int = 64
    t_sent: float = 0.0
    seq: int = 0
    t_deliver: float = 0.0


@runtime_checkable
class CommsBus(Protocol):
    def send(self, src: int, dst: int, msg: Message) -> None: ...

    def broadcast(self, src: int, msg: Message) -> None: ...

    def step(self, t: float, positions: dict[int, tuple[float, float]]) -> list[Message]: ...

    def gcs_reachable(self, t: float) -> bool: ...


# --------------------------------------------------------------------------- отказы / энергия
@runtime_checkable
class FaultManager(Protocol):
    def inject(self, event: "Event", t: float) -> None: ...

    def detect(self, t: float, agents: dict[int, "Agent"]) -> list[int]: ...


@runtime_checkable
class EnergyModel(Protocol):
    def power(self, v: float) -> float: ...


# --------------------------------------------------------------------------- метрики
@runtime_checkable
class MetricsRecorder(Protocol):
    def record(self, t: float, world: "World", agents: dict[int, "Agent"], tasks: "TaskSet", comms: Any) -> list[dict]: ...

    def finalize(self) -> dict[str, Any]: ...
