"""ВРЕМЕННЫЕ заглушки чужих модулей — только чтобы симулятор М2 запускался и тестировался.

Здесь нет «настоящих» алгоритмов М1 и моделей М3. Каждая заглушка реализует
согласованный интерфейс из ``swarmsim.interfaces`` максимально просто; владельцы
модулей заменяют их своими реализациями через реестр (``swarmsim.registry``),
ядро при этом не меняется.

| Заглушка                  | Интерфейс        | Чья настоящая реализация       |
|---------------------------|------------------|--------------------------------|
| RoundRobinStubAllocator   | Allocator        | М1 (static_*, central, CBBA…)  |
| SwathSensorStub           | Sensor (Pd)      | М3-1 модель обнаружения        |
| ConstantPowerStub         | EnergyModel      | М3-4 энергия (Zeng et al.)     |
| IdealComms / LossyCommsStub | CommsBus       | М3-2 / М3-5 CommsBus           |
| ImmediateFaultStub        | FaultManager     | М3-3 / М3-6 FaultManager       |
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from swarmsim.interfaces import GCS_ID, Assignment, Message, SwarmSnapshot

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.agents.base import Agent
    from swarmsim.eventlog import Event
    from swarmsim.missions.base import TaskSet
    from swarmsim.world import Target, World


# --------------------------------------------------------------------------- распределение
class RoundRobinStubAllocator:
    """Циклическая раздача задач по id; освобождённые — наименее загруженному живому агенту."""

    name = "stub_roundrobin"
    requires_gcs = True

    def __init__(self, **params) -> None:
        self.params = params

    def should_trigger(self, t: float, tasks: "TaskSet", pending: list) -> bool:
        return bool(pending) or bool(tasks.unassigned())

    def allocate(self, state: SwarmSnapshot, tasks: "TaskSet") -> Assignment:
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
        free = [t.id for t in tasks.unfinished() if t.id not in owned]
        initial = not owned
        for k, tid in enumerate(free):
            if initial:
                aid = alive[k % len(alive)]
                reasons[tid] = "циклическое начальное распределение"
            else:
                aid = min(alive, key=lambda i: (len(plans[i]), i))
                reasons[tid] = f"свободная задача → наименее загруженный агент {aid}"
            plans[aid].append(tid)
        return Assignment(plans=plans, reasons=reasons)


# --------------------------------------------------------------------------- сенсор
class SwathSensorStub:
    """Pd за проход = ``pd_pass``, если цель в полосе ``W/2`` от отрезка пути за шаг.

    Проекция цели берётся на полуинтервал ``[0, L)``, чтобы один прямой проход
    засчитывался один раз.
    """

    def __init__(self, swath_width: float = 40.0, pd_pass: float = 0.9, **_: object) -> None:
        self.swath_width = swath_width
        self.pd_pass = pd_pass

    def detect_prob(self, agent: "Agent", target: "Target", world: "World") -> float:
        a, b = agent.prev_state, agent.state
        if a is None or b is None or b.alt <= 0:
            return 0.0
        dx, dy = b.x - a.x, b.y - a.y
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return 0.0
        u = ((target.x - a.x) * dx + (target.y - a.y) * dy) / L
        if not (0.0 <= u < L):
            return 0.0
        d = abs((target.x - a.x) * dy - (target.y - a.y) * dx) / L
        return self.pd_pass if d <= self.swath_width / 2 else 0.0


# --------------------------------------------------------------------------- энергия
class ConstantPowerStub:
    def __init__(self, power_w: float = 250.0, **_: object) -> None:
        self.power_w = power_w

    def power(self, v: float) -> float:
        return self.power_w


# --------------------------------------------------------------------------- связь
class _OutageMixin:
    def _init_outages(self) -> None:
        self._outages: list[tuple[float, float, str]] = []

    def set_outage(self, t0: float, t1: float, scope: str) -> None:
        self._outages.append((t0, t1, scope))

    def _in_outage(self, t: float, scope: str) -> bool:
        return any(t0 <= t < t1 and (s == scope or s == "all") for t0, t1, s in self._outages)

    def gcs_reachable(self, t: float) -> bool:
        return not self._in_outage(t, "gcs")


class IdealComms(_OutageMixin):
    """Мгновенная доставка без потерь (с учётом окон потери связи и дальности)."""

    def __init__(self, range: Optional[float] = None, **_: object) -> None:
        self.range = range
        self._queue: list[Message] = []
        self._seq = 0
        self.bytes_sent = 0
        self.n_sent = 0
        self._init_outages()

    def send(self, src: int, dst: int, msg: Message) -> None:
        msg.src, msg.dst, msg.seq = src, dst, self._seq
        self._seq += 1
        self.n_sent += 1
        self.bytes_sent += msg.size
        self._queue.append(msg)

    def broadcast(self, src: int, msg: Message) -> None:
        self.send(src, None, msg)

    def _link_ok(self, t: float, src: int, dst: Optional[int], pos: dict) -> bool:
        if self._in_outage(t, "all"):
            return False
        if GCS_ID in (src, dst) and self._in_outage(t, "gcs"):
            return False
        if self.range is not None and dst is not None and src in pos and dst in pos:
            return math.dist(pos[src], pos[dst]) <= self.range
        return True

    def step(self, t: float, positions: dict[int, tuple[float, float]]) -> list[Message]:
        out = [m for m in self._queue if self._link_ok(t, m.src, m.dst, positions)]
        for m in out:
            m.t_deliver = t
        self._queue = []
        return out


class LossyCommsStub(IdealComms):
    """Задержка + дрожание + независимые потери. Порядок выдачи — по ``(t_deliver, seq)``."""

    def __init__(self, rng: np.random.Generator, delay: float = 0.2, jitter: float = 0.05,
                 loss: float = 0.1, range: Optional[float] = None, **_: object) -> None:
        super().__init__(range=range)
        self.rng, self.delay, self.jitter, self.loss = rng, delay, jitter, loss
        self._heap: list[tuple[float, int, Message]] = []
        self.n_lost = 0

    def send(self, src: int, dst: int, msg: Message) -> None:
        super().send(src, dst, msg)
        self._queue.pop()
        if self.rng.random() < self.loss:
            self.n_lost += 1
            return
        t_del = msg.t_sent + self.delay + self.jitter * float(self.rng.random())
        heapq.heappush(self._heap, (t_del, msg.seq, msg))

    def step(self, t: float, positions: dict[int, tuple[float, float]]) -> list[Message]:
        out = []
        while self._heap and self._heap[0][0] <= t + 1e-9:
            t_del, _, m = heapq.heappop(self._heap)
            if self._link_ok(t, m.src, m.dst, positions):
                m.t_deliver = t_del
                out.append(m)
            else:
                self.n_lost += 1
        return out


# --------------------------------------------------------------------------- отказы
class ImmediateFaultStub:
    """Отказ видит система через ``detect_delay`` секунд после инъекции."""

    def __init__(self, detect_delay: float = 1.0, **_: object) -> None:
        self.detect_delay = detect_delay
        self._failed_at: dict[int, float] = {}
        self._reported: set[int] = set()

    def inject(self, event: "Event", t: float) -> None:
        self._failed_at.setdefault(int(event.data["agent"]), t)

    def detect(self, t: float, agents: dict[int, "Agent"]) -> list[int]:
        out = []
        for aid in sorted(self._failed_at):
            if aid not in self._reported and t + 1e-9 >= self._failed_at[aid] + self.detect_delay:
                self._reported.add(aid)
                out.append(aid)
        return out
