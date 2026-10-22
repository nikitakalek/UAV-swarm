"""M2-1: пакет импортируется, интерфейсы на месте."""

import swarmsim
from swarmsim import interfaces


def test_package_imports():
    assert swarmsim.__version__
    for name in ("State", "Assignment", "Allocator", "Sensor", "CommsBus",
                 "FaultManager", "EnergyModel", "MetricsRecorder"):
        assert hasattr(interfaces, name)


def test_stubs_satisfy_protocols():
    from swarmsim import stubs

    assert isinstance(stubs.RoundRobinStubAllocator(), interfaces.Allocator)
    assert isinstance(stubs.SwathSensorStub(), interfaces.Sensor)
    assert isinstance(stubs.IdealComms(), interfaces.CommsBus)
    assert isinstance(stubs.ConstantPowerStub(), interfaces.EnergyModel)
    assert isinstance(stubs.ImmediateFaultStub(), interfaces.FaultManager)
