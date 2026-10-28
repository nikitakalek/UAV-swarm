"""M2-5: декомпозиция миссии поиска — число задач, покрытие 100 %, непрямоугольная область."""

import numpy as np
import pytest

from swarmsim import geometry as geo
from swarmsim.missions import decompose
from swarmsim.planning.coverage import covered_fraction, grid_shape, lawnmower
from swarmsim.scenario import MissionConfig, WorldConfig
from swarmsim.world import World


def _setup(world_kw=None, mission_kw=None):
    wc = WorldConfig.model_validate({"size": [1000, 1000], "targets": {"count": 0}, **(world_kw or {})})
    w = World.from_config(wc, np.random.default_rng(0))
    m = MissionConfig.model_validate({"swath_width": 40, **(mission_kw or {})})
    return w, m


@pytest.mark.parametrize("n", [1, 4, 9, 12, 16, 25, 7])
def test_number_of_tasks_square_area(n):
    w, m = _setup(mission_kw={"n_cells": n})
    ts = decompose(m, w)
    assert len(ts) == n
    assert len(set(ts.ids())) == n


def test_cells_tile_area_exactly():
    w, m = _setup(mission_kw={"n_cells": 16})
    ts = decompose(m, w)
    assert sum(geo.area(t.polygon) for t in ts) == pytest.approx(1_000_000)


def test_grid_shape_prefers_square_cells():
    assert grid_shape(16, 1000, 1000) == (4, 4)
    assert grid_shape(8, 2000, 1000) == (4, 2)


def test_full_coverage_square():
    w, m = _setup(mission_kw={"n_cells": 9})
    ts = decompose(m, w)
    frac = covered_fraction([t.waypoints for t in ts], w.bounds, 40, step=10)
    assert frac == pytest.approx(1.0)


def test_full_coverage_nonconvex_area():
    L = [[0, 0], [1000, 0], [1000, 400], [450, 400], [450, 1000], [0, 1000]]
    w, m = _setup(world_kw={"bounds": L}, mission_kw={"n_cells": 16})
    ts = decompose(m, w)
    # ячейки, целиком попавшие в «выемку» L, отброшены
    assert 0 < len(ts) < 16
    assert sum(geo.area(t.polygon) for t in ts) == pytest.approx(geo.area(L))
    frac = covered_fraction([t.waypoints for t in ts], L, 40, step=10)
    assert frac == pytest.approx(1.0)


def test_full_coverage_slanted_polygon():
    tri = [[0, 0], [1000, 0], [500, 900]]
    w, m = _setup(world_kw={"bounds": tri}, mission_kw={"n_cells": 9, "swath_width": 30})
    ts = decompose(m, w)
    frac = covered_fraction([t.waypoints for t in ts], tri, 30, step=10)
    assert frac == pytest.approx(1.0)


def test_sweep_avoids_nofly_and_covers_rest():
    zone = [[400, 400], [600, 400], [600, 600], [400, 600]]
    w, m = _setup(world_kw={"nofly": [zone]}, mission_kw={"n_cells": 4})
    ts = decompose(m, w)
    for t in ts:
        for p in t.waypoints:
            assert not w.in_nofly(p)
    frac = covered_fraction([t.waypoints for t in ts], w.bounds, 40, nofly=[zone], step=10)
    assert frac > 0.99


def test_lawnmower_line_spacing_not_wider_than_swath():
    wps = lawnmower(geo.rect(0, 0, 100, 130), geo.rect(0, 0, 100, 130), 40)
    ys = sorted({round(p[1], 6) for p in wps})
    assert len(ys) == 4  # ceil(130/40)
    assert all(b - a <= 40 + 1e-9 for a, b in zip(ys, ys[1:]))
    assert ys[0] <= 20 + 1e-9 and 130 - ys[-1] <= 20 + 1e-9


def test_decompose_deterministic():
    w1, m = _setup(mission_kw={"n_cells": 12})
    w2, _ = _setup(mission_kw={"n_cells": 12})
    a, b = decompose(m, w1), decompose(m, w2)
    assert [t.waypoints for t in a] == [t.waypoints for t in b]
