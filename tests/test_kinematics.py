"""M2-3: KinematicMotion — время прохождения, ограничение скорости, подменяемость."""

import math

import pytest

from swarmsim.agents.base import MotionSource
from swarmsim.agents.kinematics import KinematicMotion


def _fly(m, dt, n):
    states = [m.step(dt) for _ in range(n)]
    return states


def _armed(**kw):
    m = KinematicMotion(0, (0, 0), **kw)
    m.arm()
    m.takeoff(30)
    return m


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_reaches_waypoint_in_expected_time(dt):
    m = _armed(v_cruise=10)
    m.goto(300, 400)  # 500 м → 50 с
    t = 0.0
    while m.state().route_index < 1:
        m.step(dt)
        t += dt
        assert t < 100
    assert t == pytest.approx(50.0, abs=dt + 1e-9)
    assert m.state().pos == pytest.approx((300, 400))


def test_multi_waypoint_route_time_equals_length_over_speed():
    m = _armed(v_cruise=8)
    route = [(100, 0), (100, 100), (0, 100), (0, 0)]  # 400 м
    m.set_route(route)
    t, dt = 0.0, 0.5
    while m.state().route_index < len(route):
        m.step(dt)
        t += dt
    assert t == pytest.approx(400 / 8, abs=dt)


def test_speed_never_exceeds_vmax():
    m = _armed(v_cruise=12, v_max=12)
    m.set_route([(50, 0), (50, 3), (0, 3), (0, 6), (200, 200)])
    prev = m.state()
    for _ in range(200):
        s = m.step(0.5)
        assert s.speed <= 12 + 1e-9
        assert math.dist(prev.pos, s.pos) <= 12 * 0.5 + 1e-9
        prev = s


def test_cruise_clamped_to_vmax():
    m = _armed(v_cruise=20, v_max=15)
    assert m.v_cruise == 15


def test_not_armed_does_not_move_and_kill_stops():
    m = KinematicMotion(0, (0, 0), v_cruise=10)
    m.goto(100, 0)
    m.step(1.0)
    assert m.state().pos == (0, 0)
    m.arm()
    m.takeoff(20)
    m.step(1.0)
    assert m.state().x == pytest.approx(10)
    m.kill()
    m.step(1.0)
    assert m.state().x == pytest.approx(10)
    assert m.state().mode == "failed"


def test_is_motion_source_interface():
    """Агент работает с любым MotionSource — это то, что позволит подставить MAVSDK."""
    m = KinematicMotion(0, (0, 0))
    assert isinstance(m, MotionSource)
    for meth in ("state", "step", "set_route", "goto", "arm", "takeoff", "land", "rtl", "hold", "kill"):
        assert callable(getattr(m, meth))
