"""Разбиение площади на ячейки и «змейка» (lawnmower / boustrophedon) — M2-5.

Сознательно простая геометрия (docs/04: «сетка + змейка, интеллект — в
распределении»). Полоса обзора ``W`` задаёт шаг галсов: каждая точка ячейки
оказывается не дальше ``W/2`` от галса, что и даёт покрытие 100 %.
"""

from __future__ import annotations

import math
from typing import Sequence

from swarmsim import geometry as geo

Point = tuple[float, float]


def grid_shape(n_cells: int, width: float, height: float) -> tuple[int, int]:
    """Разложение ``n_cells = nx·ny`` с ячейками, ближайшими к квадрату."""
    if n_cells < 1:
        raise ValueError("n_cells должно быть ≥ 1")
    best = None
    for nx in range(1, n_cells + 1):
        if n_cells % nx:
            continue
        ny = n_cells // nx
        cw, ch = width / nx, height / ny
        score = abs(math.log(cw / ch))
        if best is None or score < best[0] - 1e-12:
            best = (score, nx, ny)
    return best[1], best[2]


def grid_cells(area: Sequence[Point], n_cells: int | None = None, cell_size: float | None = None
               ) -> list[tuple[int, int, list[Point]]]:
    """Прямоугольная сетка по рамке области. Возвращает ``(row, col, rect)``."""
    x0, y0, x1, y1 = geo.bbox(area)
    w, h = x1 - x0, y1 - y0
    if cell_size:
        nx, ny = max(1, math.ceil(w / cell_size - 1e-9)), max(1, math.ceil(h / cell_size - 1e-9))
    else:
        nx, ny = grid_shape(n_cells or 1, w, h)
    cw, ch = w / nx, h / ny
    cells = []
    for r in range(ny):
        for c in range(nx):
            cells.append((r, c, geo.rect(x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch)))
    return cells


def lawnmower(
    cell: Sequence[Point],
    area: Sequence[Point],
    swath: float,
    nofly: Sequence[Sequence[Point]] = (),
    min_segment: float = 1.0,
) -> list[Point]:
    """Галсы внутри ``cell ∩ area`` за вычетом бесполётных зон.

    ``cell`` — прямоугольник (ось-параллельный). Галсы горизонтальные, шаг
    ``h / ceil(h / W) ≤ W``; каждый галс покрывает свою полосу целиком по x.
    """
    if swath <= 0:
        raise ValueError("ширина полосы должна быть > 0")
    cx0, cy0, cx1, cy1 = geo.bbox(cell)
    h = cy1 - cy0
    n_lines = max(1, math.ceil(h / swath - 1e-9))
    s = h / n_lines
    pts: list[Point] = []
    direction = 1
    for k in range(n_lines):
        ylo, yhi = cy0 + k * s, cy0 + (k + 1) * s
        y = 0.5 * (ylo + yhi)
        iv = [(max(a, cx0), min(b, cx1)) for a, b in geo.band_intervals(area, ylo, yhi)]
        iv = [(a, b) for a, b in iv if b - a > 1e-9]
        if not iv:
            continue
        cut: list[tuple[float, float]] = []
        for z in nofly:
            cut.extend(geo.band_intervals(z, ylo, yhi))
        iv = geo.subtract_intervals(iv, cut, min_len=min_segment) if cut else iv
        if not iv:
            continue
        if direction < 0:
            iv = [(b, a) for a, b in reversed(iv)]
        for a, b in iv:
            if pts and math.hypot(pts[-1][0] - a, pts[-1][1] - y) < 1e-9:
                pts.append((b, y))
            else:
                pts.extend([(a, y), (b, y)])
        direction = -direction
    return pts


def covered_fraction(
    segments_points: Sequence[Sequence[Point]],
    area: Sequence[Point],
    swath: float,
    nofly: Sequence[Sequence[Point]] = (),
    step: float = 5.0,
) -> float:
    """Доля свободных точек области (сетка с шагом ``step``) в пределах ``W/2`` от пути."""
    x0, y0, x1, y1 = geo.bbox(area)
    segs = []
    for pts in segments_points:
        segs.extend((pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    half = swath / 2 + 1e-6
    total = hit = 0
    ny, nx = int((y1 - y0) / step), int((x1 - x0) / step)
    for i in range(ny):
        y = y0 + (i + 0.5) * step
        for j in range(nx):
            x = x0 + (j + 0.5) * step
            p = (x, y)
            if not geo.point_in_polygon(p, area) or any(geo.point_in_polygon(p, z) for z in nofly):
                continue
            total += 1
            for a, b in segs:
                if min(a[1], b[1]) - half <= y <= max(a[1], b[1]) + half:
                    if geo.point_segment_distance(p, a, b) <= half:
                        hit += 1
                        break
    return hit / total if total else 1.0
