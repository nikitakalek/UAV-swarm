"""2D-геометрия без внешних зависимостей: полигоны, отрезки, сканирующие прямые.

Полигон — список вершин ``[(x, y), ...]`` без повтора первой вершины, обход любой.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]

EPS = 1e-9


def area(poly: Polygon) -> float:
    """Площадь (без знака) по формуле шнурования."""
    return abs(signed_area(poly))


def signed_area(poly: Polygon) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def centroid(poly: Polygon) -> Point:
    a = signed_area(poly)
    if abs(a) < EPS:
        xs, ys = zip(*poly)
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = x1 * y2 - x2 * y1
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    return (cx / (6 * a), cy / (6 * a))


def bbox(poly: Polygon) -> tuple[float, float, float, float]:
    xs, ys = zip(*poly)
    return (min(xs), min(ys), max(xs), max(ys))


def rect(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def point_on_boundary(p: Point, poly: Polygon, tol: float = 1e-7) -> bool:
    n = len(poly)
    for i in range(n):
        if point_segment_distance(p, poly[i], poly[(i + 1) % n]) <= tol:
            return True
    return False


def point_in_polygon(p: Point, poly: Polygon, boundary: bool = True) -> bool:
    """Чётно-нечётное правило; ``boundary`` — считать ли границу внутренней."""
    if point_on_boundary(p, poly):
        return boundary
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_properly_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Пересечение отрезков во внутренней точке обоих (касание и коллинеарность — нет)."""
    d1 = _orient(c, d, a)
    d2 = _orient(c, d, b)
    d3 = _orient(a, b, c)
    d4 = _orient(a, b, d)
    tol = 1e-9 * max(1.0, abs(d1), abs(d2), abs(d3), abs(d4))
    return ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and (
        (d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)
    )


def segment_intersects_polygon(a: Point, b: Point, poly: Polygon) -> bool:
    """Заходит ли отрезок во внутренность полигона (касание границы не считается)."""
    n = len(poly)
    for i in range(n):
        if segments_properly_intersect(a, b, poly[i], poly[(i + 1) % n]):
            return True
    # отрезок целиком внутри или проходит через внутренность между вершинами
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if point_in_polygon(q, poly, boundary=False):
            return True
    return False


def polygons_overlap(p: Polygon, q: Polygon, samples: int = 40) -> bool:
    """Есть ли у полигонов общая внутренняя область (общая граница — не перекрытие).

    Точные проверки (пересечение рёбер, вершина внутри) дополняются выборкой точек
    в пересечении рамок — для случаев с совпадающими рёбрами. Перекрытия тоньше
    шага выборки (≈ 1/``samples`` рамки) могут быть не замечены.
    """
    n, m = len(p), len(q)
    for i in range(n):
        for j in range(m):
            if segments_properly_intersect(p[i], p[(i + 1) % n], q[j], q[(j + 1) % m]):
                return True
    for a, b in ((p, q), (q, p)):
        for v in a:
            if point_in_polygon(v, b, boundary=False):
                return True
        if point_in_polygon(interior_point(a), b, boundary=False):
            return True
    # совпадающие/коллинеарные границы: проверка выборкой в пересечении рамок
    px0, py0, px1, py1 = bbox(p)
    qx0, qy0, qx1, qy1 = bbox(q)
    x0, y0, x1, y1 = max(px0, qx0), max(py0, qy0), min(px1, qx1), min(py1, qy1)
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        return False
    k = samples
    for i in range(k):
        y = y0 + (i + 0.5) * (y1 - y0) / k
        for j in range(k):
            pt = (x0 + (j + 0.5) * (x1 - x0) / k, y)
            if point_in_polygon(pt, p, boundary=False) and point_in_polygon(pt, q, boundary=False):
                return True
    return False


def polygon_inside(inner: Polygon, outer: Polygon) -> bool:
    """``inner`` целиком внутри ``outer`` (касание границы допускается)."""
    if not all(point_in_polygon(v, outer, boundary=True) for v in inner):
        return False
    n, m = len(inner), len(outer)
    for i in range(n):
        for j in range(m):
            if segments_properly_intersect(inner[i], inner[(i + 1) % n], outer[j], outer[(j + 1) % m]):
                return False
    return True


def interior_point(poly: Polygon) -> Point:
    """Точка строго внутри полигона (центроид, если подходит, иначе середина скан-интервала)."""
    c = centroid(poly)
    if point_in_polygon(c, poly, boundary=False):
        return c
    x0, y0, x1, y1 = bbox(poly)
    for k in range(1, 20):
        y = y0 + (y1 - y0) * k / 20
        iv = scanline_intervals(poly, y)
        if iv:
            a, b = max(iv, key=lambda ab: ab[1] - ab[0])
            if b - a > EPS:
                return ((a + b) / 2, y)
    return c


def clip_polygon(subject: Polygon, clip: Polygon) -> list[Point]:
    """Сазерленд–Ходжман: ``subject`` (любой) ∩ ``clip`` (выпуклый)."""
    if signed_area(clip) < 0:
        clip = list(reversed(clip))
    out = list(subject)
    m = len(clip)
    for i in range(m):
        if not out:
            break
        a, b = clip[i], clip[(i + 1) % m]
        inp, out = out, []
        n = len(inp)
        for k in range(n):
            cur, prev = inp[k], inp[k - 1]
            cur_in = _orient(a, b, cur) >= -EPS
            prev_in = _orient(a, b, prev) >= -EPS
            if cur_in:
                if not prev_in:
                    out.append(_line_cross(prev, cur, a, b))
                out.append(cur)
            elif prev_in:
                out.append(_line_cross(prev, cur, a, b))
    return _dedupe(out)


def _line_cross(p: Point, q: Point, a: Point, b: Point) -> Point:
    x1, y1 = p
    x2, y2 = q
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < EPS:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _dedupe(pts: list[Point]) -> list[Point]:
    res: list[Point] = []
    for p in pts:
        if not res or math.hypot(p[0] - res[-1][0], p[1] - res[-1][1]) > 1e-9:
            res.append(p)
    if len(res) > 1 and math.hypot(res[0][0] - res[-1][0], res[0][1] - res[-1][1]) <= 1e-9:
        res.pop()
    return res


def scanline_intervals(poly: Polygon, y: float) -> list[tuple[float, float]]:
    """Интервалы ``x`` пересечения горизонтали ``y`` с внутренностью полигона."""
    xs: list[float] = []
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    xs.sort()
    return [(xs[k], xs[k + 1]) for k in range(0, len(xs) - 1, 2) if xs[k + 1] - xs[k] > EPS]


def merge_intervals(iv: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    res: list[list[float]] = []
    for a, b in sorted(iv):
        if res and a <= res[-1][1] + EPS:
            res[-1][1] = max(res[-1][1], b)
        else:
            res.append([a, b])
    return [(a, b) for a, b in res]


def subtract_intervals(
    base: list[tuple[float, float]], cut: list[tuple[float, float]], min_len: float = 0.0
) -> list[tuple[float, float]]:
    res = []
    cut = merge_intervals(cut)
    for a, b in base:
        cur = a
        for c, d in cut:
            if d <= cur or c >= b:
                continue
            if c > cur:
                res.append((cur, c))
            cur = max(cur, d)
        if cur < b:
            res.append((cur, b))
    return [(a, b) for a, b in res if b - a > max(min_len, EPS)]


def band_intervals(poly: Polygon, y_lo: float, y_hi: float) -> list[tuple[float, float]]:
    """Проекция на ось x части полигона в полосе ``[y_lo, y_hi]``.

    Экстремумы по x у кусочно-линейной границы достигаются на краях полосы или в
    вершинах, поэтому достаточно объединить сечения на этих уровнях и в середине.
    """
    levels = {y_lo + 1e-7, y_hi - 1e-7, 0.5 * (y_lo + y_hi)}
    for x, y in poly:
        if y_lo < y < y_hi:
            levels.add(y + 1e-7)
            levels.add(y - 1e-7)
    iv: list[tuple[float, float]] = []
    for y in sorted(levels):
        iv.extend(scanline_intervals(poly, y))
    return merge_intervals(iv)


def path_length(points: Sequence[Point]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))
