"""Деление участка на компактные секторы равной площади (М1).

Конвейер :func:`build_partition`: решётка семян → Ллойд (компактность) → power diagram
(взвешенный Вороной, точное равенство площадей). Разброс площадей 1–2 %, вытянутость
секторов ≈ 1 — в отличие от полос, меньше холостых перелётов и суммарная длина границ.

Все функции детерминированы и зависят только от геометрии.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, voronoi_diagram
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def _clean(geom: BaseGeometry | None) -> BaseGeometry:
    """Приводит геометрию сектора к непустой Polygon/MultiPolygon или к пустому Polygon."""
    empty = Polygon()
    if geom is None or geom.is_empty:
        return empty
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not parts:
            return empty
        geom = unary_union(parts)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        return empty
    return geom if geom.area > 0 else empty


def _safe_intersection(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    """Пересечение, устойчивое к численно вырожденной геометрии.

    Ячейки Вороного после нескольких итераций Ллойда могут стать почти
    вырожденными, и GEOS бросает ``TopologyException``. В этом случае
    повторяем операцию на исправленной геометрии.
    """
    try:
        return a.intersection(b)
    except Exception:
        try:
            return make_valid(a).intersection(make_valid(b))
        except Exception:
            return Polygon()


@dataclass(frozen=True)
class SectorPartition:
    """Результат разбиения области на секторы.

    ``sectors[i]`` соответствует ``seeds[i]``. Пустые секторы допустимы
    (например, при бесполётных зонах) — они просто не получат задач.
    """

    sectors: tuple[BaseGeometry, ...]
    seeds: tuple[tuple[float, float], ...]

    def __len__(self) -> int:
        return len(self.sectors)

    @property
    def areas(self) -> np.ndarray:
        return np.array([s.area for s in self.sectors], dtype=float)

    def total_area(self) -> float:
        return float(self.areas.sum())


def area_stats(partition: SectorPartition) -> dict[str, float]:
    """Статистика площадей секторов — нужна для отчёта ."""
    a = np.array([s.area for s in partition.sectors if not s.is_empty], dtype=float)
    if a.size == 0:
        return {"n": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "ratio": float("inf"), "cv": float("nan")}
    return {
        "n": float(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "ratio": float(a.max() / a.min()) if a.min() > 0 else float("inf"),
        "cv": float(a.std() / a.mean()) if a.mean() > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Диаграмма Вороного
# ---------------------------------------------------------------------------


def grid_seeds(domain: BaseGeometry, n: int, lattice: str = "grid") -> list[tuple[float, float]]:
    """Ставит ``n`` семян внутри домена.

    ``lattice="grid"`` — квадратная решётка (по умолчанию).
    ``lattice="hex"``  — гексагональная решётка: даёт соты, теоретически
    оптимальную упаковку (минимум суммарного периметра).

    Для прямоугольного домена точки попадают точно в узлы решётки. Если часть
    узлов выпадает (домен с бесполётными зонами), из попавших внутрь точек
    детерминированно выбираются ``n`` самых разнесённых.
    """
    if n < 1:
        raise ValueError("n должно быть >= 1")
    if lattice not in ("grid", "hex"):
        raise ValueError("lattice должен быть 'grid' или 'hex'")

    minx, miny, maxx, maxy = domain.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        raise ValueError("домен вырожден (нулевая площадь габаритов)")

    pts = _lattice_points(minx, miny, w, h, n, lattice)
    inside = [p for p in pts if domain.covers(Point(p))]
    if len(inside) >= n:
        # обычный случай: все узлы внутри — берём ровно n в порядке решётки
        return pts[:n] if len(inside) == len(pts) else _spread_sample(inside, n)

    for mult in (2, 4, 8):
        pts = _lattice_points(minx, miny, w, h, n * mult, lattice)
        inside = [p for p in pts if domain.covers(Point(p))]
        if len(inside) >= n:
            return _spread_sample(inside, n)

    raise ValueError(
        f"не удалось поставить {n} семян внутри домена: область слишком узкая "
        f"или сильно вырезана бесполётными зонами"
    )


def _lattice_points(
    minx: float, miny: float, w: float, h: float, n: int, lattice: str
) -> list[tuple[float, float]]:
    """Узлы решётки над габаритным прямоугольником — ровно ``n`` точек.

    Число точек в ряду распределяется равномерно: когда ``n`` не делится на
    число рядов, «хвост» размазывается по рядам, а не сваливается в последний
    ряд. Иначе начальное положение семян сильно неравномерно и Ллойд
    сходится к плохому разбиению.
    """
    aspect = w / h

    if lattice == "hex":
        cols = max(1, int(np.ceil((n * aspect * 2 / np.sqrt(3)) ** 0.5)))
        rows = max(1, int(np.ceil(n / cols)))
        base, rem = divmod(n, rows)
        dx, dy = w / cols, h / rows
        x_edge = minx + w - 1e-9
        pts: list[tuple[float, float]] = []
        for r in range(rows):
            cnt = base + (1 if r < rem else 0)
            if cnt <= 0:
                continue
            y = miny + (r + 0.5) * dy
            for c in range(cnt):
                x = minx + (c + 0.5 + (0.5 if r % 2 else 0.0)) * dx
                pts.append((float(min(x, x_edge)), float(y)))
        return pts

    cols = max(1, int(np.ceil((n * aspect) ** 0.5)))
    rows = max(1, int(np.ceil(n / cols)))
    base, rem = divmod(n, rows)
    pts = []
    for r in range(rows):
        cnt = base + (1 if r < rem else 0)
        if cnt <= 0:
            continue
        y = miny + (r + 0.5) * (h / rows)
        xs = minx + (np.arange(cnt) + 0.5) * (w / cnt)
        pts.extend((float(x), float(y)) for x in xs)
    return pts


def _spread_sample(points: Sequence[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Детерминированно выбирает ``n`` самых разнесённых точек (farthest-point sampling)."""
    pts = list(points)
    if len(pts) < n:
        raise ValueError("точек меньше, чем требуется")
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    first = min(range(len(pts)), key=lambda i: (pts[i][0] - cx) ** 2 + (pts[i][1] - cy) ** 2)
    chosen = [pts[first]]
    dist = [math.dist(p, pts[first]) for p in pts]
    while len(chosen) < n:
        i = max(range(len(pts)), key=lambda k: dist[k])
        chosen.append(pts[i])
        dist = [min(dist[k], math.dist(pts[k], pts[i])) for k in range(len(pts))]
    return chosen


def voronoi_partition(domain: BaseGeometry, seeds: Sequence[tuple[float, float]]) -> SectorPartition:
    """Строит ячейки Вороного для ``seeds`` и обрезает их по ``domain``."""
    seeds = [(float(x), float(y)) for x, y in seeds]
    if not seeds:
        return SectorPartition((), ())

    envelope = box(*domain.bounds)
    diagram = voronoi_diagram(MultiPoint(seeds), envelope=envelope)

    cells: list[BaseGeometry] = []
    for sx, sy in seeds:
        p = Point(sx, sy)
        cell = None
        for geom in diagram.geoms:
            if geom.contains(p):
                cell = geom
                break
        if cell is None:  # численный краевой случай: берём ближайшую ячейку
            cell = min(diagram.geoms, key=lambda g: g.distance(p))
        cells.append(_clean(_safe_intersection(cell, domain)))

    return SectorPartition(tuple(cells), tuple(seeds))


def lloyd_equalize(
    domain: BaseGeometry,
    seeds: Sequence[tuple[float, float]],
    iters: int = 50,
    tol: float = 1e-6,
) -> SectorPartition:
    """Выравнивает площади секторов алгоритмом Ллойда (CVT).

    На каждой итерации семя сдвигается в центроид своей ячейки. Это делает
    ячейки компактнее и выравнивает их площади *приблизительно* (точное
    равенство даёт только power diagram).
    """
    seeds = [(float(x), float(y)) for x, y in seeds]
    for _ in range(max(0, iters)):
        part = voronoi_partition(domain, seeds)
        moved: list[tuple[float, float]] = []
        max_move = 0.0
        for (sx, sy), cell in zip(seeds, part.sectors):
            if cell.is_empty:
                moved.append((sx, sy))
                continue
            c = cell.centroid
            nx, ny = float(c.x), float(c.y)
            max_move = max(max_move, float(np.hypot(nx - sx, ny - sy)))
            moved.append((nx, ny))
        seeds = moved
        if max_move < tol:
            break
    return voronoi_partition(domain, seeds)


# ---------------------------------------------------------------------------
# Power diagram (взвешенный Вороной) — точное равенство площадей
# ---------------------------------------------------------------------------


def _halfplane_polygon(
    a: np.ndarray, b: float, domain: BaseGeometry
) -> BaseGeometry | None:
    """Прямоугольник, накрывающий полуплоскость ``a·x <= b`` рядом с доменом."""
    norm = math.hypot(float(a[0]), float(a[1]))
    if norm <= 0.0:
        return None
    nx, ny = float(a[0]) / norm, float(a[1]) / norm
    c = float(b) / norm
    minx, miny, maxx, maxy = domain.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    r = 2.0 * (diag + abs(c)) + 1.0
    px, py = nx * c, ny * c
    tx, ty = -ny, nx
    return Polygon(
        [
            (px + tx * r, py + ty * r),
            (px + tx * r - nx * r, py + ty * r - ny * r),
            (px - tx * r - nx * r, py - ty * r - ny * r),
            (px - tx * r, py - ty * r),
        ]
    )


def power_partition(
    domain: BaseGeometry,
    sites: Sequence[tuple[float, float]],
    weights: Sequence[float] | None = None,
    neighbor_limit: int | None = None,
) -> SectorPartition:
    """Ячейки power diagram (взвешенного Вороного).

    Ячейка точки ``i``: ``{ x : |x − p_i|² − w_i ≤ |x − p_j|² − w_j  для всех j }``.

    Отличие от обычного Вороного: веса. Площадь ячейки **монотонно растёт** с её
    весом, поэтому подбором весов можно добиться точного равенства площадей при
    сохранении компактной формы (см. :func:`power_equalize`).

    ``neighbor_limit`` ограничивает число полуплоскостей на ячейку: клиппинг идёт
    по ближайшим соседям. Для небольших N ограничение не действует.
    """
    sites = [(float(x), float(y)) for x, y in sites]
    n = len(sites)
    if n == 0:
        return SectorPartition((), ())
    if weights is None:
        weights = [0.0] * n
    weights = [float(w) for w in weights]
    if neighbor_limit is None:
        neighbor_limit = n - 1 if n <= 24 else 24

    pts = np.array(sites, dtype=float)
    sq = (pts ** 2).sum(axis=1)

    cells: list[BaseGeometry] = []
    for i in range(n):
        cell: BaseGeometry = domain
        order = np.argsort(((pts - pts[i]) ** 2).sum(axis=1))
        clipped = 0
        for j in order:
            if int(j) == i:
                continue
            if clipped >= neighbor_limit:
                break
            a = 2.0 * (pts[j] - pts[i])
            b = float(sq[j] - sq[i] + weights[i] - weights[j])
            halfplane = _halfplane_polygon(a, b, domain)
            if halfplane is None:
                continue
            cell = _safe_intersection(cell, halfplane)
            clipped += 1
            if cell.is_empty:
                break
        cells.append(_clean(cell))

    return SectorPartition(tuple(cells), tuple(sites))


def power_equalize(
    domain: BaseGeometry,
    seeds: Sequence[tuple[float, float]],
    iters: int = 60,
    damping: float = 0.5,
    tol: float = 0.01,
    neighbor_limit: int | None = None,
) -> SectorPartition:
    """Подбирает веса power diagram так, чтобы площади секторов совпали.

    Площадь ячейки монотонно растёт с весом, поэтому используем простую итерацию
    ``w_i += damping · (A_target − A_i)``. Веса центрируются: power diagram не
    меняется от прибавления константы ко всем весам.
    """
    sites = [(float(x), float(y)) for x, y in seeds]
    n = len(sites)
    if n == 0:
        return SectorPartition((), ())

    target = domain.area / n
    weights = [0.0] * n
    partition = power_partition(domain, sites, weights, neighbor_limit=neighbor_limit)

    for _ in range(max(0, iters)):
        areas = np.array([c.area for c in partition.sectors], dtype=float)
        error = target - areas
        if np.max(np.abs(error)) / target < tol:
            break
        weights = [w + damping * float(e) for w, e in zip(weights, error)]
        mean_w = sum(weights) / n
        weights = [w - mean_w for w in weights]
        partition = power_partition(
            domain, sites, weights, neighbor_limit=neighbor_limit
        )

    return partition


def build_partition(
    domain: BaseGeometry,
    n: int,
    lattice: str = "grid",
    lloyd_iters: int = 50,
    power_iters: int = 60,
) -> SectorPartition:
    """Полный конвейер разбиения области на ``n`` секторов.

    1. решётка семян (``lattice``);
    2. Ллойд — компактность и равномерность семян (``lloyd_iters``);
    3. power diagram — **точное равенство площадей** (``power_iters``).

    ``power_iters=0`` отключает третий шаг (остаётся обычный Вороной + Ллойд).
    """
    seeds = grid_seeds(domain, n, lattice=lattice)
    if lloyd_iters > 0:
        seeds = list(lloyd_equalize(domain, seeds, iters=lloyd_iters).seeds)
    if power_iters <= 0:
        return voronoi_partition(domain, seeds)
    return power_equalize(domain, seeds, iters=power_iters)
