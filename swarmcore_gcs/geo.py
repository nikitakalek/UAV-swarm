"""Широта/долгота ↔ локальные метры (x — восток, y — север) вокруг точки отсчёта.

Равнопромежуточная проекция: на участках до нескольких километров ошибка — сантиметры,
этого достаточно для планирования облёта поля.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

R_EARTH = 6378137.0


@dataclass(frozen=True)
class LocalFrame:
    lat0: float
    lon0: float

    @property
    def _kx(self) -> float:
        return math.radians(1.0) * R_EARTH * math.cos(math.radians(self.lat0))

    @property
    def _ky(self) -> float:
        return math.radians(1.0) * R_EARTH

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return ((lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky)

    def to_ll(self, x: float, y: float) -> tuple[float, float]:
        return (self.lat0 + y / self._ky, self.lon0 + x / self._kx)

    def poly_xy(self, pts: Iterable) -> list[tuple[float, float]]:
        return [tuple(round(v, 3) for v in self.to_xy(p[0], p[1])) for p in pts]

    def poly_ll(self, pts: Iterable) -> list[list[float]]:
        return [[round(v, 7) for v in self.to_ll(p[0], p[1])] for p in pts]
