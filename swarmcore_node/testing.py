"""LocalSwarm — N узлов и наземная станция в одном процессе с общим модельным временем.

Узлы общаются только через ``MemoryHub`` (тот же протокол, что по UDP), аппараты —
``KinematicMotion``. Нужен для быстрых детерминированных тестов логики узла: отказы,
смена лидера, потери сообщений, разрыв связи с наземной станцией.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Optional

from swarmcore_node.gcs import GroundStation
from swarmcore_node.link import MemoryHub
from swarmcore_node.messages import GCS_ID
from swarmcore_node.node import NodeConfig, SwarmNode
from swarmsim.agents.kinematics import KinematicMotion
from swarmsim.scenario import Scenario


class LocalSwarm:
    def __init__(self, scenario: Scenario | dict, config: Optional[NodeConfig] = None, *,
                 loss: float = 0.0, seed: int = 0, dt: float = 0.1) -> None:
        self.sc = scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario)
        self.hub = MemoryHub(loss=loss, seed=seed)
        self.dt = dt
        self.t = 0.0
        ag = self.sc.agents
        self.nodes = [
            SwarmNode(i, KinematicMotion(i, ag.home_of(i), v_cruise=ag.speed), self.hub.link(i), config)
            for i in range(ag.count)
        ]
        self.gcs = GroundStation(self.hub.link(GCS_ID), self.sc, mission_id="test")
        self._schedule: list[tuple[float, Callable[[], None]]] = []
        self.fail_times: dict[int, float] = {}

    # ------------------------------------------------------------------ сценарий
    def at(self, t: float, fn: Callable[[], None]) -> None:
        self._schedule.append((t, fn))
        self._schedule.sort(key=lambda x: x[0])

    def kill(self, i: int) -> None:
        self.fail_times[i] = self.t
        self.nodes[i].crash()
        self.hub.down.add(i)

    def run(self, until: float = 3600.0) -> "LocalSwarm":
        while self.t < until:
            while self._schedule and self._schedule[0][0] <= self.t:
                self._schedule.pop(0)[1]()
            self.gcs.step(self.t)
            for n in self.nodes:
                n.step(self.t)
            if self.finished():
                break
            self.t = round(self.t + self.dt, 9)
        return self

    def finished(self) -> bool:
        return all(n.crashed or n.finished for n in self.nodes) and any(n.finished for n in self.nodes)

    # ------------------------------------------------------------------ итоги
    def summary(self) -> dict[str, Any]:
        live = [n for n in self.nodes if not n.crashed]
        n_tasks = len(live[0].tasks) if live and live[0].tasks is not None else 0
        done_by = Counter(e.data["task"] for n in self.nodes for e in n.log.of_type("task_done"))
        done_known = set().union(*({t.id for t in n.tasks if t.is_finished} for n in live)) if live else set()
        replans = [(n.id, e) for n in self.nodes for e in n.log.of_type("plan_published")]
        realloc = {}
        for i, tf in self.fail_times.items():
            after = [e.t for _, e in replans if e.t >= tf and i in e.data.get("gone", [])]
            realloc[i] = round(min(after) - tf, 2) if after else None
        return {
            "n_tasks": n_tasks,
            "done": len(done_known),
            "lost": n_tasks - len(done_known),
            "duplicates": sum(1 for c in done_by.values() if c > 1),
            "realloc_s": realloc,
            "t_end": self.t,
            "modes": {n.id: n.mode.value for n in self.nodes},
            "leaders": sorted({nid for nid, _ in replans}),
        }
