"""Тесты геометрии секторов (M1-1)."""

from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon, box

from swarmsim.sectors import (
    area_stats,
    build_partition,
    grid_seeds,
    lloyd_equalize,
    power_equalize,
    power_partition,
    strip_partition,
    voronoi_partition,
)

N_VALUES = [5, 10, 20]


@pytest.fixture
def domain():
    """Область 1000 x 1000 м."""
    return box(0.0, 0.0, 1000.0, 1000.0)


# --- наивные полосы (B1) ---------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
def test_strip_partition_equal_area(domain, n):
    part = strip_partition(domain, n)
    assert len(part) == n
    a = part.areas
    assert a.min() > 0
    assert a.max() / a.min() < 1.01, "полосы должны быть равны по площади"
    assert a.sum() == pytest.approx(domain.area, rel=1e-6)


@pytest.mark.parametrize("n", N_VALUES)
def test_strip_partition_axis_y(domain, n):
    part = strip_partition(domain, n, axis="y")
    assert len(part) == n
    a = part.areas
    assert a.max() / a.min() < 1.01


def test_strip_partition_rejects_bad_axis(domain):
    with pytest.raises(ValueError):
        strip_partition(domain, 3, axis="z")


def test_strip_partition_single(domain):
    part = strip_partition(domain, 1)
    assert len(part) == 1
    assert part.total_area() == pytest.approx(domain.area, rel=1e-6)


# --- семена ----------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("lattice", ["grid", "hex"])
def test_seeds_inside_domain(domain, n, lattice):
    seeds = grid_seeds(domain, n, lattice=lattice)
    assert len(seeds) == n
    for x, y in seeds:
        assert domain.covers(Point(x, y)), f"семя ({x}, {y}) вне домена"


def test_seeds_unique(domain):
    seeds = grid_seeds(domain, 20)
    assert len(set(seeds)) == 20


def test_seeds_in_domain_with_hole():
    """Домен с вырезанной бесполётной зоной: семена не должны попадать в дыру."""
    outer = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    hole = [(300, 300), (700, 300), (700, 700), (300, 700)]
    dom = Polygon(outer, [hole])
    seeds = grid_seeds(dom, 12)
    assert len(seeds) == 12
    for x, y in seeds:
        assert dom.covers(Point(x, y))


# --- Вороной ---------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
def test_voronoi_covers_domain(domain, n):
    part = voronoi_partition(domain, grid_seeds(domain, n))
    assert len(part) == n
    assert part.total_area() == pytest.approx(domain.area, rel=1e-6)
    assert all(not s.is_empty for s in part.sectors)


@pytest.mark.parametrize("n", [5, 10])
def test_voronoi_sectors_do_not_overlap(domain, n):
    part = lloyd_equalize(domain, grid_seeds(domain, n), iters=30)
    for i in range(len(part.sectors)):
        for j in range(i + 1, len(part.sectors)):
            inter = part.sectors[i].intersection(part.sectors[j]).area
            assert inter == pytest.approx(0.0, abs=1e-6)


# --- выравнивание (CVT / Ллойд) -------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
def test_lloyd_does_not_worsen_area_equality(domain, n):
    seeds = grid_seeds(domain, n)
    before = area_stats(voronoi_partition(domain, seeds))
    after = area_stats(lloyd_equalize(domain, seeds, iters=100))
    assert after["ratio"] <= before["ratio"] + 1e-9


@pytest.mark.parametrize("n", N_VALUES)
def test_lloyd_area_dispersion(domain, n):
    """Разброс площадей после выравнивания — измеряем и держим в разумных рамках."""
    after = area_stats(lloyd_equalize(domain, grid_seeds(domain, n), iters=200))
    print(f"\nn={n}: ratio={after['ratio']:.3f} cv={after['cv']:.3f}")
    assert after["ratio"] < 1.6, f"площади разъехались сильнее ожидаемого: {after}"


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("lattice", ["grid", "hex"])
def test_lloyd_partition_is_valid(domain, n, lattice):
    part = lloyd_equalize(domain, grid_seeds(domain, n, lattice=lattice), iters=50)
    assert part.total_area() == pytest.approx(domain.area, rel=1e-6)
    assert all(not s.is_empty for s in part.sectors)


def test_determinism(domain):
    p1 = lloyd_equalize(domain, grid_seeds(domain, 10), iters=30)
    p2 = lloyd_equalize(domain, grid_seeds(domain, 10), iters=30)
    assert p1.seeds == p2.seeds
    for s1, s2 in zip(p1.sectors, p2.sectors):
        assert s1.equals(s2)


# --- крайние случаи --------------------------------------------------------


def test_single_sector_is_whole_domain(domain):
    part = lloyd_equalize(domain, grid_seeds(domain, 1), iters=10)
    assert len(part) == 1
    assert part.total_area() == pytest.approx(domain.area, rel=1e-6)


def test_narrow_domain():
    """Узкая область: решётка должна подстроиться, а не сломаться."""
    dom = box(0.0, 0.0, 1000.0, 50.0)
    seeds = grid_seeds(dom, 10)
    assert len(seeds) == 10
    part = lloyd_equalize(dom, seeds, iters=50)
    assert part.total_area() == pytest.approx(dom.area, rel=1e-6)


def test_zero_seeds_returns_empty(domain):
    part = voronoi_partition(domain, [])
    assert len(part) == 0


# --- power diagram: точное равенство площадей ------------------------------


@pytest.mark.parametrize("n", N_VALUES)
def test_power_partition_covers_domain(domain, n):
    seeds = lloyd_equalize(domain, grid_seeds(domain, n), iters=50).seeds
    part = power_partition(domain, seeds)
    assert len(part) == n
    assert part.total_area() == pytest.approx(domain.area, rel=1e-6)
    assert all(not s.is_empty for s in part.sectors)


@pytest.mark.parametrize("n", N_VALUES)
def test_power_equalize_makes_areas_equal(domain, n):
    """Главный смысл power diagram: площади почти точно равны (в отличие от CVT)."""
    seeds = lloyd_equalize(domain, grid_seeds(domain, n), iters=50).seeds
    stats = area_stats(power_equalize(domain, seeds, iters=100))
    print(f"\nn={n}: power ratio={stats['ratio']:.4f} cv={stats['cv']:.4f}")
    assert stats["ratio"] < 1.05, f"площади не выровнялись: {stats}"


@pytest.mark.parametrize("n", N_VALUES)
def test_power_equalize_better_than_cvt(domain, n):
    """Там, где CVT даёт разброс, power diagram обязан его убрать.

    На N=20 сетка 4×5 точно замощает квадрат, CVT уже идеален — тогда
    достаточно, чтобы power diagram не был хуже.
    """
    seeds = grid_seeds(domain, n)
    cvt = area_stats(lloyd_equalize(domain, seeds, iters=50))
    power = area_stats(
        power_equalize(domain, lloyd_equalize(domain, seeds, iters=50).seeds, iters=100)
    )
    assert power["ratio"] <= cvt["ratio"] + 1e-9, (
        f"power diagram ({power['ratio']:.3f}) хуже CVT ({cvt['ratio']:.3f})"
    )
    if cvt["ratio"] > 1.05:
        assert power["ratio"] < cvt["ratio"], (
            f"power diagram ({power['ratio']:.3f}) не улучшил CVT ({cvt['ratio']:.3f})"
        )


def test_power_sectors_do_not_overlap(domain):
    seeds = lloyd_equalize(domain, grid_seeds(domain, 10), iters=50).seeds
    part = power_equalize(domain, seeds, iters=100)
    for i in range(len(part.sectors)):
        for j in range(i + 1, len(part.sectors)):
            inter = part.sectors[i].intersection(part.sectors[j]).area
            assert inter == pytest.approx(0.0, abs=1e-6)


def test_power_determinism(domain):
    seeds = lloyd_equalize(domain, grid_seeds(domain, 10), iters=50).seeds
    p1 = power_equalize(domain, seeds, iters=60)
    p2 = power_equalize(domain, seeds, iters=60)
    for s1, s2 in zip(p1.sectors, p2.sectors):
        assert s1.equals(s2)


@pytest.mark.parametrize("n", N_VALUES)
def test_build_partition_pipeline(domain, n):
    """Полный конвейер: решётка → Ллойд → power diagram."""
    part = build_partition(domain, n)
    assert len(part) == n
    assert part.total_area() == pytest.approx(domain.area, rel=1e-6)
    stats = area_stats(part)
    assert stats["ratio"] < 1.05


def test_build_partition_without_power(domain):
    """power_iters=0 должен возвращать обычный Вороной + Ллойд."""
    part = build_partition(domain, 10, power_iters=0)
    assert len(part) == 10
    assert area_stats(part)["ratio"] > 1.05  # без выравнивания площади разъезжаются


def test_power_equalize_keeps_compactness(domain):
    """Равные площади не должны достигаться ценой вытянутых секторов."""
    part = build_partition(domain, 10)
    q = max(s.length / (4.0 * (s.area ** 0.5)) for s in part.sectors if not s.is_empty)
    assert q < 1.4, f"секторы стали вытянутыми (q={q:.2f})"


def test_power_equalize_on_domain_with_hole():
    outer = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    hole = [(300, 300), (700, 300), (700, 700), (300, 700)]
    dom = Polygon(outer, [hole])
    seeds = lloyd_equalize(dom, grid_seeds(dom, 12), iters=50).seeds
    part = power_equalize(dom, seeds, iters=100)
    assert part.total_area() == pytest.approx(dom.area, rel=1e-4)
