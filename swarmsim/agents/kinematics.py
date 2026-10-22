"""KinematicMotion (M2-3): безмассовая кинематическая модель (docs/02, §3.1, MVP-0).

``p(t+dt) = p(t) + v_c · normalize(wp − p(t)) · dt`` с переносом остатка пути на
следующие точки маршрута внутри одного шага, поэтому время прохождения маршрута
равно ``длина / v_c`` с точностью до одного шага и не зависит от числа точек.
"""

from __future__ import annotations

import math
from typing import Sequence

from swarmsim.agents.base import MotionSource
from swarmsim.interfaces import State


class KinematicMotion(MotionSource):
    def __init__(
        self,
        agent_id: int,
        start: tuple[float, float],
        v_cruise: float = 10.0,
        v_max: float | None = None,
        altitude: float = 40.0,
        accept_radius: float = 1e-6,
    ) -> None:
        if v_cruise <= 0:
            raise ValueError("v_cruise должна быть > 0")
        self.agent_id = agent_id
        self.v_max = v_max if v_max is not None else v_cruise
        self.v_cruise = min(v_cruise, self.v_max)
        self.home = (float(start[0]), float(start[1]))
        self.x, self.y = self.home
        self.alt_target = altitude
        self.alt = 0.0
        self.vx = self.vy = 0.0
        self.t = 0.0
        self.armed = False
        self.mode = "landed"
        self.accept_radius = accept_radius
        self._route: list[tuple[float, float]] = []
        self._idx = 0
        self._killed = False

    # ------------------------------------------------------------------ команды
    def arm(self) -> None:
        if not self._killed:
            self.armed = True

    def takeoff(self, alt: float | None = None) -> None:
        if not self.armed or self._killed:
            return
        self.alt = self.alt_target if alt is None else alt
        self.mode = "idle"

    def set_route(self, points: Sequence[tuple[float, float]]) -> None:
        if self._killed:
            return
        self._route = [(float(x), float(y)) for x, y in points]
        self._idx = 0

    def hold(self) -> None:
        self._route = []
        self._idx = 0

    def land(self) -> None:
        self.hold()
        self.alt = 0.0
        self.mode = "landed"

    def rtl(self) -> None:
        self.set_route([self.home])

    def kill(self) -> None:
        self.hold()
        self._killed = True
        self.mode = "failed"

    # ------------------------------------------------------------------ шаг
    def step(self, dt: float) -> State:
        x0, y0 = self.x, self.y
        budget = self.v_cruise * dt if (self.armed and self.mode != "landed" and not self._killed) else 0.0
        while budget > 0 and self._idx < len(self._route):
            tx, ty = self._route[self._idx]
            d = math.hypot(tx - self.x, ty - self.y)
            if d <= budget + self.accept_radius:
                self.x, self.y = tx, ty
                budget -= d
                self._idx += 1
            else:
                self.x += (tx - self.x) / d * budget
                self.y += (ty - self.y) / d * budget
                budget = 0.0
        self.t += dt
        self.vx, self.vy = (self.x - x0) / dt, (self.y - y0) / dt
        if not self._killed and self.mode != "landed":
            self.mode = "flying" if self._idx < len(self._route) else "idle"
        return self.state()

    def state(self) -> State:
        return State(
            agent_id=self.agent_id, t=self.t, x=self.x, y=self.y, alt=self.alt,
            vx=self.vx, vy=self.vy, armed=self.armed, mode=self.mode, route_index=self._idx,
        )
