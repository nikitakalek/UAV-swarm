"""M2-11: MavsdkMotion взаимозаменяем с KinematicMotion (проверка интерфейса без SITL)."""

import os

import pytest

from swarmsim.agents.base import MotionSource
from swarmsim.agents.kinematics import KinematicMotion


def test_same_interface_as_kinematic():
    from swarmcore_px4.mavsdk_motion import MavsdkMotion, sitl_address

    m = MavsdkMotion(2, (8.0, 0.0), connect=False)
    try:
        assert isinstance(m, MotionSource)
        assert m.address == sitl_address(2) == "udpin://0.0.0.0:14542"
        assert m.grpc_port == 50053
        k = KinematicMotion(2, (8.0, 0.0))
        public = {n for n in dir(MotionSource) if not n.startswith("_")}
        for name in public:
            assert callable(getattr(m, name)) and callable(getattr(k, name))
        s = m.state()  # без телеметрии — позиция дома
        assert (s.x, s.y) == (8.0, 0.0)
        m.set_route([(10, 10)])
        assert m.state().mode == "flying"
    finally:
        m._loop.call_soon_threadsafe(m._loop.stop)


@pytest.mark.sitl
@pytest.mark.skipif(not os.environ.get("SWARMCORE_SITL"), reason="нужен запущенный PX4 SITL (SWARMCORE_SITL=1)")
def test_sitl_check_two_vehicles():
    from swarmcore_px4.run_sitl import main

    assert main(["check", "--n", "2"]) == 0
