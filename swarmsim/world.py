"""World (M2-2): двумерная среда — границы, бесполётные зоны, цели, карта вероятности.

Все случайные величины берутся из генератора, переданного при создании, поэтому
при одном зерне мир (позиции целей, исходы обнаружения) полностью детерминирован.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np

from swarmsim import geometry as geo
from swarmsim.interfaces import Detection, Hit, Sensor

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.agents.base import Agent
    from swarmsim.scenario import WorldConfig


@dataclass
class Target:
    id: int
    x: float
    y: float
    t_appear: float = 0.0
    priority: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    p_cum: float = 0.0  # накопленная вероятность обнаружения 1 − Π(1 − p_k)
    detected_t: Optional[float] = None
    detected_by: Optional[int] = None

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class ProbabilityMap:
    """Априорная карта вероятности нахождения цели на регулярной сетке."""

    x0: float
    y0: float
    cell: float
    p: np.ndarray  # shape (ny, nx), сумма = 1

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        ny, nx = self.p.shape
        i = min(max(int((y - self.y0) // self.cell), 0), ny - 1)
        j = min(max(int((x - self.x0) // self.cell), 0), nx - 1)
        return i, j

    def value(self, x: float, y: float) -> float:
        i, j = self.cell_of(x, y)
        return float(self.p[i, j])


@dataclass
class World:
    bounds: list[tuple[float, float]]
    nofly: list[list[tuple[float, float]]] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)
    prob_map: Optional[ProbabilityMap] = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    t: float = 0.0
    nofly_margin: float = 5.0

    # ------------------------------------------------------------------ построение
    @classmethod
    def from_config(cls, cfg: "WorldConfig", rng: np.random.Generator) -> "World":
        bounds = [tuple(p) for p in cfg.bounds_polygon()]
        nofly = [[tuple(p) for p in z] for z in cfg.nofly]
        w = cls(bounds=bounds, nofly=nofly, rng=rng, nofly_margin=cfg.nofly_margin)
        w.prob_map = w._build_prob_map(cfg)
        tc = cfg.targets
        if tc.positions:
            for p in tc.positions:
                w.add_target(p[0], p[1])
        else:
            for _ in range(tc.count):
                x, y = w.sample_target_position()
                w.add_target(x, y)
        return w

    def _build_prob_map(self, cfg: "WorldConfig") -> ProbabilityMap:
        x0, y0, x1, y1 = geo.bbox(self.bounds)
        c = cfg.prob_map_cell
        nx, ny = max(1, math.ceil((x1 - x0) / c)), max(1, math.ceil((y1 - y0) / c))
        xs = x0 + (np.arange(nx) + 0.5) * c
        ys = y0 + (np.arange(ny) + 0.5) * c
        X, Y = np.meshgrid(xs, ys)
        tc = cfg.targets
        if tc.distribution == "clusters" and tc.clusters:
            p = np.zeros_like(X)
            for cl in tc.clusters:
                cx, cy = cl.center
                p += cl.weight * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * cl.sigma**2))
        else:
            p = np.ones_like(X)
        mask = np.array(
            [[self.is_free((float(X[i, j]), float(Y[i, j]))) for j in range(nx)] for i in range(ny)]
        )
        p = np.where(mask, p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.full_like(p, 1.0 / p.size)
        return ProbabilityMap(x0=x0, y0=y0, cell=c, p=p)

    def sample_target_position(self) -> tuple[float, float]:
        pm = self.prob_map
        assert pm is not None
        flat = pm.p.ravel()
        ny, nx = pm.p.shape
        for _ in range(1000):
            k = int(self.rng.choice(flat.size, p=flat))
            i, j = divmod(k, nx)
            x = pm.x0 + (j + self.rng.random()) * pm.cell
            y = pm.y0 + (i + self.rng.random()) * pm.cell
            if self.is_free((x, y)):
                return (float(x), float(y))
        raise RuntimeError("не удалось разместить цель: свободная площадь пуста")

    def add_target(self, x: float, y: float, t_appear: Optional[float] = None, priority: float = 1.0) -> Target:
        tgt = Target(id=len(self.targets), x=float(x), y=float(y), priority=priority,
                     t_appear=self.t if t_appear is None else t_appear)
        self.targets.append(tgt)
        return tgt

    # ------------------------------------------------------------------ геометрия
    def contains(self, p: tuple[float, float]) -> bool:
        return geo.point_in_polygon(p, self.bounds)

    def in_nofly(self, p: tuple[float, float]) -> bool:
        return any(geo.point_in_polygon(p, z, boundary=False) for z in self.nofly)

    def is_free(self, p: tuple[float, float]) -> bool:
        return self.contains(p) and not self.in_nofly(p)

    def segment_free(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        return not any(geo.segment_intersects_polygon(a, b, z) for z in self.nofly)

    def free_path(self, a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]]:
        """Кратчайший обход бесполётных зон по графу видимости углов их рамок.

        Возвращает точки после ``a`` до ``b`` включительно.
        """
        if self.segment_free(a, b):
            return [b]
        m = self.nofly_margin
        nodes: list[tuple[float, float]] = [a, b]
        for z in self.nofly:
            x0, y0, x1, y1 = geo.bbox(z)
            for c in geo.rect(x0 - m, y0 - m, x1 + m, y1 + m):
                if not self.in_nofly(c):
                    nodes.append(c)
        n = len(nodes)
        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == 1:
                break
            for v in range(n):
                if v == u:
                    continue
                if not self.segment_free(nodes[u], nodes[v]):
                    continue
                nd = d + math.dist(nodes[u], nodes[v])
                if nd < dist[v] - 1e-12:
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        if math.isinf(dist[1]):
            return [b]
        path = []
        v = 1
        while v != 0:
            path.append(nodes[v])
            v = prev[v]
        return list(reversed(path))

    # ------------------------------------------------------------------ динамика
    def step(self, dt: float) -> None:
        """Продвигает модельное время и (если заданы) движущиеся цели."""
        for tg in self.targets:
            if tg.vx or tg.vy:
                nx, ny = tg.x + tg.vx * dt, tg.y + tg.vy * dt
                if self.is_free((nx, ny)):
                    tg.x, tg.y = nx, ny
                else:
                    tg.vx, tg.vy = -tg.vx, -tg.vy
        self.t += dt

    def active_targets(self) -> list[Target]:
        return [tg for tg in self.targets if tg.t_appear <= self.t + 1e-9]

    def observe(self, agent: "Agent", sensor: Sensor) -> Detection:
        """Одна попытка обнаружения всех активных целей сенсором агента.

        Вероятность ``p`` даёт модель сенсора (М3); мир накапливает ``p_cum`` и
        разыгрывает исход своим генератором — порядок вызовов фиксирован циклом.
        """
        det = Detection(t=self.t, agent_id=agent.id)
        if not agent.alive:
            return det
        for tg in self.active_targets():
            p = float(sensor.detect_prob(agent, tg, self))
            if p <= 0.0:
                continue
            p = min(p, 1.0)
            tg.p_cum = 1.0 - (1.0 - tg.p_cum) * (1.0 - p)
            hit = False
            if tg.detected_t is None:
                hit = bool(self.rng.random() < p)
                if hit:
                    tg.detected_t, tg.detected_by = self.t, agent.id
            det.hits.append(Hit(target_id=tg.id, p=p, detected=hit))
        return det

    # ------------------------------------------------------------------ вывод
    def snapshot(self) -> dict:
        return {
            "t": self.t,
            "bounds": [list(p) for p in self.bounds],
            "nofly": [[list(p) for p in z] for z in self.nofly],
            "targets": [
                {"id": tg.id, "x": tg.x, "y": tg.y, "t_appear": tg.t_appear, "priority": tg.priority,
                 "p_cum": tg.p_cum, "detected_t": tg.detected_t, "detected_by": tg.detected_by}
                for tg in self.targets
            ],
        }

    def free_area(self) -> float:
        return geo.area(self.bounds) - sum(geo.area(z) for z in self.nofly)


def polygons_from(raw: Iterable[Sequence[Sequence[float]]]) -> list[list[tuple[float, float]]]:
    return [[(float(p[0]), float(p[1])) for p in poly] for poly in raw]
