"""Схема сценария (M2-7): pydantic + YAML.

Сценарий описывает область, число агентов, цели и расписание событий. Всё, что
влияет на результат, лежит в сценарии — отсюда хэш конфига в ``run_manifest.json``.
Версия схемы — ``schema_version``; несовместимые изменения повышают её.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from swarmsim import geometry as geo

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.world import World

SCHEMA_VERSION = 1

Point = tuple[float, float]
PolygonT = list[Point]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_polygon(poly: PolygonT, what: str) -> PolygonT:
    if len(poly) < 3:
        raise ValueError(f"{what}: полигон должен иметь ≥ 3 вершин")
    if geo.area(poly) <= 0:
        raise ValueError(f"{what}: полигон нулевой площади")
    return poly


# --------------------------------------------------------------------------- мир
class Cluster(_Model):
    center: Point
    sigma: float = Field(100.0, gt=0)
    weight: float = Field(1.0, gt=0)


class TargetsConfig(_Model):
    count: int = Field(20, ge=0)
    distribution: Literal["uniform", "clusters"] = "uniform"
    clusters: list[Cluster] = Field(default_factory=list)
    positions: list[Point] = Field(default_factory=list, description="явные координаты вместо розыгрыша")


class WorldConfig(_Model):
    size: Optional[Point] = Field((1000.0, 1000.0), description="прямоугольник [0,W]×[0,H], если нет bounds")
    bounds: Optional[PolygonT] = None
    nofly: list[PolygonT] = Field(default_factory=list)
    nofly_margin: float = Field(5.0, ge=0)
    prob_map_cell: float = Field(50.0, gt=0)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)

    def bounds_polygon(self) -> PolygonT:
        if self.bounds:
            return list(self.bounds)
        w, h = self.size
        return geo.rect(0.0, 0.0, w, h)

    @model_validator(mode="after")
    def _validate(self) -> "WorldConfig":
        b = _check_polygon(self.bounds_polygon(), "world.bounds")
        for k, z in enumerate(self.nofly):
            _check_polygon(z, f"world.nofly[{k}]")
            if not any(geo.point_in_polygon(v, b) for v in z):
                raise ValueError(f"world.nofly[{k}] целиком вне границ мира")
        return self


# --------------------------------------------------------------------------- агенты / сенсор
class AgentsConfig(_Model):
    count: int = Field(5, ge=1)
    base: Point = (0.0, 0.0)
    homes: list[Point] = Field(default_factory=list, description="явные точки старта (иначе — ряд у base)")
    spacing: float = Field(5.0, ge=0)
    speed: float = Field(10.0, gt=0, description="крейсерская скорость v_c, м/с")
    v_max: Optional[float] = Field(None, gt=0)
    altitude: float = Field(40.0, gt=0)
    altitude_step: float = Field(5.0, ge=0, description="эшелонирование по высоте между агентами")
    min_separation: float = Field(3.0, gt=0)

    @model_validator(mode="after")
    def _homes(self) -> "AgentsConfig":
        if self.homes and len(self.homes) != self.count:
            raise ValueError("agents.homes: число точек должно равняться agents.count")
        if self.v_max is not None and self.speed > self.v_max:
            raise ValueError("agents.speed > agents.v_max")
        return self

    def home_of(self, i: int) -> Point:
        if self.homes:
            return tuple(self.homes[i])
        return (self.base[0] + i * self.spacing, self.base[1])


class SensorConfig(_Model):
    model: str = Field("stub_swath", description="имя модели сенсора в реестре (М3 подменяет)")
    swath_width: float = Field(40.0, gt=0)
    pd_pass: float = Field(0.9, gt=0, le=1)
    params: dict[str, Any] = Field(default_factory=dict)


class EnergyConfig(_Model):
    model: str = "stub_constant"
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- миссия
class SectorConfig(_Model):
    """Сектор, размеченный пользователем вручную (M2-13)."""

    id: str
    polygon: PolygonT
    agents: list[int] = Field(default_factory=list, description="кому отдать сектор (пусто — решит распределитель)")

    @field_validator("polygon")
    @classmethod
    def _poly(cls, v: PolygonT) -> PolygonT:
        return _check_polygon(v, "sector.polygon")


class MissionConfig(_Model):
    type: Literal["search"] = "search"
    area: Optional[PolygonT] = Field(None, description="область поиска (иначе — границы мира)")
    n_cells: Optional[int] = Field(16, ge=1)
    cell_size: Optional[float] = Field(None, gt=0)
    swath_width: Optional[float] = Field(None, gt=0, description="шаг галсов; по умолчанию = sensor.swath_width")
    min_cell_area: float = Field(1.0, ge=0)
    sectors_version: int = Field(1, description="версия формата секторов (M2-13)")
    sectors: list[SectorConfig] = Field(default_factory=list)
    sector_assignment: Literal["centroid", "overlap", "clip"] = "centroid"

    def area_polygon(self, world: "World") -> PolygonT:
        return list(self.area) if self.area else list(world.bounds)


# --------------------------------------------------------------------------- события
class _EventBase(_Model):
    t: float = Field(..., ge=0, description="модельное время, с")


class AgentFailureEvent(_EventBase):
    type: Literal["agent_failure"]
    agent: int = Field(..., ge=0)


class NewTargetEvent(_EventBase):
    type: Literal["new_target"]
    pos: Point
    priority: float = 1.0
    search_radius: float = Field(0.0, ge=0, description=">0 — добавить задачу доразведки вокруг точки")


class CommsLossEvent(_EventBase):
    type: Literal["comms_loss"]
    duration: float = Field(..., gt=0)
    scope: Literal["gcs", "all"] = Field("gcs", description="gcs — нет связи с наземной станцией; all — нет никакой")


Event = Annotated[Union[AgentFailureEvent, NewTargetEvent, CommsLossEvent], Field(discriminator="type")]


class RandomFailures(_Model):
    fraction: float = Field(0.0, ge=0, le=1, description="доля агентов, отказывающих за миссию (λ)")
    t_window: tuple[float, float] = (0.0, 0.0)


class FaultsConfig(_Model):
    model: str = "stub_immediate"
    detect_delay: float = Field(1.0, ge=0, description="задержка детекции отказа (тайм-аут heartbeat), с")
    random_failures: RandomFailures = Field(default_factory=RandomFailures)


class CommsConfig(_Model):
    model: str = Field("ideal", description="ideal | stub_lossy | имя модели М3 в реестре")
    range: Optional[float] = Field(None, gt=0, description="R_comm, м (None — без ограничения)")
    delay: float = Field(0.0, ge=0)
    jitter: float = Field(0.0, ge=0)
    loss: float = Field(0.0, ge=0, le=1)
    heartbeat_period: float = Field(1.0, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)


class AllocatorConfig(_Model):
    name: str = "stub_roundrobin"
    params: dict[str, Any] = Field(default_factory=dict)


class OutputConfig(_Model):
    log_every: float = Field(1.0, gt=0, description="шаг записи траекторий, с")


# --------------------------------------------------------------------------- сценарий
class Scenario(_Model):
    schema_version: int = SCHEMA_VERSION
    name: str = "scenario"
    description: str = ""
    dt: float = Field(0.5, gt=0)
    t_end: float = Field(3600.0, gt=0)
    stop_when_done: bool = True
    world: WorldConfig = Field(default_factory=WorldConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    energy: EnergyConfig = Field(default_factory=EnergyConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    allocator: AllocatorConfig = Field(default_factory=AllocatorConfig)
    comms: CommsConfig = Field(default_factory=CommsConfig)
    faults: FaultsConfig = Field(default_factory=FaultsConfig)
    events: list[Event] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version={v} не поддерживается (ожидается {SCHEMA_VERSION})")
        return v

    @model_validator(mode="after")
    def _cross(self) -> "Scenario":
        if self.mission.swath_width is None:
            self.mission.swath_width = self.sensor.swath_width
        for ev in self.events:
            if isinstance(ev, AgentFailureEvent) and ev.agent >= self.agents.count:
                raise ValueError(f"событие agent_failure: агента {ev.agent} нет (count={self.agents.count})")
            if ev.t > self.t_end:
                raise ValueError(f"событие {ev.type} в t={ev.t} позже t_end={self.t_end}")
        if self.mission.sectors:
            from swarmsim.missions.sectors import validate_sectors

            validate_sectors(self)
        return self

    # ------------------------------------------------------------------ ввод/вывод
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.from_yaml(path)
