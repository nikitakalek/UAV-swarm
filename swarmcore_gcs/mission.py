"""Черновик миссии с карты пульта → сценарий для узлов и предпросмотр плана.

Черновик (JSON из браузера), координаты — ``[широта, долгота]``::

    {"area": [[lat, lon], ...], "nofly": [[[lat, lon], ...], ...], "homes": [[lat, lon], ...],
     "params": {"altitude": 15, "altitude_step": 4, "speed": 5, "swath": 20, "cell_size": 50,
                "battery_wh": 200, "time_scale": 5}}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import ValidationError
from shapely.geometry import Polygon

from swarmcore_gcs.geo import LocalFrame
from swarmsim.alloc import SectorAllocator
from swarmsim.interfaces import AgentView, SwarmSnapshot
from swarmsim.missions import decompose
from swarmsim.planning.validator import PlanValidator
from swarmsim.scenario import Scenario
from swarmsim.world import World

DEFAULTS = {"altitude": 15.0, "altitude_step": 4.0, "speed": 5.0, "swath": 20.0, "cell_size": 50.0,
            "battery_wh": 200.0, "time_scale": 5.0, "power_w": 250.0}
WORLD_MARGIN = 150.0  # м вокруг участка и площадок: запас для участков, добавленных в полёте


class DraftError(ValueError):
    """Черновик нельзя превратить в миссию — текст для оператора."""


@dataclass
class Mission:
    draft: dict[str, Any]
    params: dict[str, float]
    frame: LocalFrame
    scenario: Scenario
    updates: list[dict[str, Any]] = field(default_factory=list)  # ревизии в полёте (метры и градусы)

    @property
    def n(self) -> int:
        return self.scenario.agents.count

    def altitudes(self) -> dict[int, float]:
        a = self.scenario.agents
        return {i: a.altitude + i * a.altitude_step for i in range(a.count)}

    def geometry(self) -> dict[str, Any]:
        """Геометрия для пульта и 3D: в градусах и в метрах."""
        from swarmcore_node.mission_update import apply_update

        sc = self.scenario
        world = World.from_config(sc.world, np.random.default_rng(0))
        tasks = decompose(sc.mission, world)
        for u in self.updates:
            apply_update(world, tasks, sc.mission, u)
        return {
            "origin": [self.frame.lat0, self.frame.lon0],
            "area": self.draft["area"], "homes": self.draft["homes"],
            "nofly": self.draft.get("nofly", []) + [z for u in self.updates for z in u["nofly_ll"]],
            "areas_extra": [a for u in self.updates for a in u["areas_ll"]],
            "areas_extra_xy": [a for u in self.updates for a in u["areas"]],
            "rev": len(self.updates),
            "bounds_xy": [list(p) for p in world.bounds],
            "area_xy": [list(p) for p in sc.mission.area],
            "nofly_xy": [[list(p) for p in z] for z in world.nofly],
            "homes_xy": [list(sc.agents.home_of(i)) for i in range(self.n)],
            "altitudes": [self.altitudes()[i] for i in range(self.n)],
            "params": self.params,
            "cells": [{"id": t.id, "poly": self.frame.poly_ll(t.polygon), "poly_xy": [list(p) for p in t.polygon],
                       "wps_xy": [[round(x, 2), round(y, 2)] for x, y in t.waypoints],
                       "cancelled": t.status.value == "failed"} for t in tasks],
        }


def _pts(raw, what: str, min_len: int = 3) -> list[list[float]]:
    try:
        pts = [[float(p[0]), float(p[1])] for p in raw]
    except (TypeError, ValueError, IndexError):
        raise DraftError(f"{what}: ожидаются точки [широта, долгота]") from None
    if len(pts) < min_len:
        raise DraftError(f"{what}: нужно не меньше {min_len} точек")
    return pts


def build(draft: dict[str, Any]) -> Mission:
    area = _pts(draft.get("area") or [], "Участок")
    homes = _pts(draft.get("homes") or [], "Площадки старта", min_len=1)
    nofly = [_pts(z, f"Бесполётная зона {k + 1}") for k, z in enumerate(draft.get("nofly") or [])]
    params = {**DEFAULTS, **{k: float(v) for k, v in (draft.get("params") or {}).items() if v not in (None, "")}}
    if len(homes) > 10:
        raise DraftError("Не больше 10 аппаратов")
    if params["altitude_step"] < 3:
        raise DraftError("Шаг эшелонов меньше 3 м: аппараты на соседних высотах не будут разнесены")

    frame = LocalFrame(*homes[0])
    area_xy = frame.poly_xy(area)
    if not Polygon(area_xy).is_valid:
        raise DraftError("Участок пересекает сам себя — перерисуйте контур")
    if Polygon(area_xy).area < 100:
        raise DraftError("Участок меньше 100 м²")
    if Polygon(area_xy).area > 4e6:
        raise DraftError("Участок больше 4 км² — для демо возьмите меньше")
    homes_xy = frame.poly_xy(homes)
    nofly_xy = [frame.poly_xy(z) for z in nofly]
    xs = [p[0] for p in area_xy + homes_xy] + [p[0] for z in nofly_xy for p in z]
    ys = [p[1] for p in area_xy + homes_xy] + [p[1] for z in nofly_xy for p in z]
    m = WORLD_MARGIN
    bounds = [(min(xs) - m, min(ys) - m), (max(xs) + m, min(ys) - m), (max(xs) + m, max(ys) + m),
              (min(xs) - m, max(ys) + m)]
    data = {
        "name": "gcs", "description": "Миссия с пульта оператора",
        "world": {"bounds": bounds, "nofly": nofly_xy, "targets": {"count": 0}},
        "agents": {"count": len(homes), "homes": homes_xy, "speed": params["speed"],
                   "altitude": params["altitude"], "altitude_step": params["altitude_step"]},
        "sensor": {"swath_width": params["swath"]},
        "mission": {"area": area_xy, "n_cells": None, "cell_size": params["cell_size"]},
        "allocator": {"name": "sector"},
    }
    try:
        sc = Scenario.model_validate(data)
    except ValidationError as e:
        raise DraftError("; ".join(err["msg"] for err in e.errors())) from None
    return Mission({"area": area, "nofly": nofly, "homes": homes}, params, frame, sc)


def make_update(mission: Mission, body: dict[str, Any]) -> dict[str, Any]:
    """Изменение миссии с карты: ``{"nofly": [[lat, lon], ...]}`` или ``{"area": [[lat, lon], ...]}``."""
    out: dict[str, Any] = {"nofly": [], "areas": [], "nofly_ll": [], "areas_ll": []}
    bounds = Polygon(mission.scenario.world.bounds_polygon())
    for key, what in (("nofly", "Бесполётная зона"), ("area", "Участок")):
        if not body.get(key):
            continue
        ll = _pts(body[key], what)
        xy = mission.frame.poly_xy(ll)
        poly = Polygon(xy)
        if not poly.is_valid or poly.area < 25:
            raise DraftError(f"{what}: контур пересекает сам себя или меньше 25 м²")
        if key == "area" and not bounds.contains(poly):
            raise DraftError(f"Участок выходит за район миссии (до {WORLD_MARGIN:.0f} м от исходной разметки)")
        if key == "nofly" and not bounds.intersects(poly):
            raise DraftError("Зона целиком вне района миссии — она ни на что не влияет")
        out["nofly" if key == "nofly" else "areas"].append([list(p) for p in xy])
        out["nofly_ll" if key == "nofly" else "areas_ll"].append(ll)
    if not out["nofly"] and not out["areas"]:
        raise DraftError("Нарисуйте зону или участок")
    return out


def preview(mission: Mission) -> dict[str, Any]:
    """План до взлёта: секторы, маршруты, время, энергия, замечания валидатора."""
    sc, p = mission.scenario, mission.params
    world = World.from_config(sc.world, np.random.default_rng(0))
    tasks = decompose(sc.mission, world)
    if not len(tasks):
        raise DraftError("В участке не получилось ни одной ячейки: участок целиком в бесполётной зоне?")
    agents = {i: AgentView(agent_id=i, pos=sc.agents.home_of(i), home=sc.agents.home_of(i), alive=True,
                           speed=sc.agents.speed, current_task=None, queue=[], energy_used_wh=0.0)
              for i in range(mission.n)}
    snap = SwarmSnapshot(t=0.0, agents=agents, world=world)
    asg = SectorAllocator().allocate(snap, tasks)
    report = PlanValidator(battery_wh=p["battery_wh"], power_w=p["power_w"]).validate(
        asg, snap, tasks, mission.altitudes())
    owner = asg.owner_of()
    per_agent = []
    for i in range(mission.n):
        path = report.paths.get(i, ())
        length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        t = length / sc.agents.speed
        per_agent.append({"id": i, "altitude": mission.altitudes()[i], "tasks": len(asg.plans.get(i, [])),
                          "length_m": round(length), "time_s": round(t),
                          "energy_wh": round(p["power_w"] * t / 3600, 1),
                          "path": mission.frame.poly_ll(path)})
    return {
        "ok": report.ok, "violations": [str(v) for v in report.violations],
        "area_m2": round(Polygon(sc.mission.area).area), "n_tasks": len(tasks),
        "time_s": max((a["time_s"] for a in per_agent), default=0),
        "agents": per_agent,
        "cells": [{"id": t.id, "poly": mission.frame.poly_ll(t.polygon), "owner": owner.get(t.id)} for t in tasks],
    }
