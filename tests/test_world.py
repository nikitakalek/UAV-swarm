"""M2-2: World — границы, бесполётные зоны, цели, детерминизм."""

import numpy as np
import pytest

from swarmsim import geometry as geo
from swarmsim.scenario import WorldConfig
from swarmsim.world import World


def _world(seed=1, **kw):
    cfg = WorldConfig.model_validate({"size": [1000, 1000], **kw})
    return World.from_config(cfg, np.random.default_rng(seed))


def test_point_inside_nofly_detected():
    w = _world(nofly=[[[100, 100], [300, 100], [300, 300], [100, 300]]])
    assert w.in_nofly((200, 200))
    assert not w.in_nofly((50, 50))
    assert not w.in_nofly((100, 200)), "граница зоны — не внутренность"
    assert not w.is_free((200, 200))
    assert w.is_free((500, 500))


def test_point_inside_nonconvex_nofly():
    # L-образная зона: точка в «выемке» — снаружи
    L = [[0, 0], [400, 0], [400, 100], [100, 100], [100, 400], [0, 400]]
    w = _world(nofly=[L])
    assert w.in_nofly((50, 300))
    assert w.in_nofly((300, 50))
    assert not w.in_nofly((300, 300))


def test_targets_avoid_nofly_and_bounds():
    zone = [[200, 200], [800, 200], [800, 800], [200, 800]]
    w = _world(nofly=[zone], targets={"count": 200})
    assert len(w.targets) == 200
    for t in w.targets:
        assert w.contains(t.pos)
        assert not w.in_nofly(t.pos)


def test_world_deterministic_by_seed():
    a, b, c = _world(3, targets={"count": 30}), _world(3, targets={"count": 30}), _world(4, targets={"count": 30})
    assert [t.pos for t in a.targets] == [t.pos for t in b.targets]
    assert [t.pos for t in a.targets] != [t.pos for t in c.targets]


def test_clustered_targets_follow_probability_map():
    w = _world(targets={"count": 300, "distribution": "clusters",
                        "clusters": [{"center": [200, 200], "sigma": 60}]})
    near = sum(geo.point_segment_distance(t.pos, (200, 200), (200, 200)) < 200 for t in w.targets)
    assert near / 300 > 0.9
    assert w.prob_map.p.sum() == pytest.approx(1.0)


def test_free_path_avoids_nofly():
    zone = [[400, 0], [600, 0], [600, 800], [400, 800]]
    w = _world(nofly=[zone])
    a, b = (100, 400), (900, 400)
    path = [a] + w.free_path(a, b)
    assert path[-1] == b
    for p, q in zip(path, path[1:]):
        assert w.segment_free(p, q)
    assert geo.path_length(path) > 800


class _Probe:
    def __init__(self, p):
        self.p = p
        self.swath_width = 10

    def detect_prob(self, agent, target, world):
        return self.p


class _A:
    id = 0
    alive = True


def test_observe_accumulates_pd_and_is_deterministic():
    def go(seed):
        w = _world(seed, targets={"count": 5})
        s = _Probe(0.3)
        hits = []
        for _ in range(5):
            d = w.observe(_A(), s)
            hits.append(tuple(d.new_detections))
            w.step(1.0)
        return w, hits

    w1, h1 = go(7)
    w2, h2 = go(7)
    assert h1 == h2
    assert w1.targets[0].p_cum == pytest.approx(1 - 0.7 ** 5)
    assert w1.t == pytest.approx(5.0)
