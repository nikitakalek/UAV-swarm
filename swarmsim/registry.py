"""Реестр подключаемых реализаций: по имени из сценария → объект.

Имя — либо зарегистрированный псевдоним, либо путь ``"package.module:Class"``.
Модули М1/М3 подключаются, не трогая ядро: ``register_allocator("cbba",
"swarmsim.alloc.cbba:CBBAAllocator")`` или прямо ``allocator.name:
swarmsim.alloc.cbba:CBBAAllocator`` в YAML.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

ALLOCATORS: dict[str, str] = {
    "stub_roundrobin": "swarmsim.stubs:RoundRobinStubAllocator",
    "manual_sectors": "swarmsim.missions.sectors:ManualSectorAllocator",
    # М1 через адаптер к симулятору (swarmsim/m1_bridge.py)
    "static_naive": "swarmsim.m1_bridge:StaticNaiveSim",
    "static_voronoi": "swarmsim.m1_bridge:StaticVoronoiSim",
    "central_greedy": "swarmsim.m1_bridge:CentralGreedySim",
}
SENSORS: dict[str, str] = {"stub_swath": "swarmsim.stubs:SwathSensorStub"}
ENERGY: dict[str, str] = {"stub_constant": "swarmsim.stubs:ConstantPowerStub"}
COMMS: dict[str, str] = {"ideal": "swarmsim.stubs:IdealComms", "stub_lossy": "swarmsim.stubs:LossyCommsStub"}
FAULTS: dict[str, str] = {"stub_immediate": "swarmsim.stubs:ImmediateFaultStub"}


def _resolve(table: dict[str, str], name: str, kind: str) -> Callable[..., Any]:
    path = table.get(name, name)
    if ":" not in path:
        raise KeyError(f"{kind} '{name}' не зарегистрирован; доступны: {sorted(table)} или 'module:Class'")
    mod, attr = path.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as e:
        raise KeyError(f"{kind} '{name}' → '{path}' не найден: {e}") from e


def register_allocator(name: str, path: str) -> None:
    ALLOCATORS[name] = path


def make_allocator(name: str, **params: Any):
    return _resolve(ALLOCATORS, name, "allocator")(**params)


def make_sensor(name: str, **params: Any):
    return _resolve(SENSORS, name, "sensor")(**params)


def make_energy(name: str, **params: Any):
    return _resolve(ENERGY, name, "energy")(**params)


def make_comms(name: str, **params: Any):
    return _resolve(COMMS, name, "comms")(**params)


def make_faults(name: str, **params: Any):
    return _resolve(FAULTS, name, "faults")(**params)
