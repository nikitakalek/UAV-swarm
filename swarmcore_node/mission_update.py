"""Изменение миссии в полёте: новые бесполётные зоны и дополнительные участки.

Изменение — ревизия ``{"rev": N, "nofly": [полигон, ...], "areas": [полигон, ...]}`` в
локальных метрах. Каждый узел применяет ревизии по порядку одной и той же функцией, поэтому
копии миссии на бортах совпадают без обмена задачами:

* новая зона: у незавершённых ячеек, которые она задевает, галсы строятся заново в обход
  зоны (прогресс ячейки сбрасывается); ячейка целиком внутри зоны отменяется;
* новый участок: нарезается на ячейки того же размера, id — ``u<rev>_r..._c...``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from shapely.geometry import LineString, Polygon

from swarmsim import geometry as geo
from swarmsim.missions.base import Task, TaskSet, TaskStatus
from swarmsim.planning.coverage import grid_cells, lawnmower

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.scenario import MissionConfig
    from swarmsim.world import World


def _cell_size(mission: "MissionConfig", tasks: TaskSet) -> float:
    if mission.cell_size:
        return mission.cell_size
    areas = [geo.area(t.polygon) for t in tasks if len(t.polygon) >= 3]
    return math.sqrt(sum(areas) / len(areas)) if areas else 50.0


def apply_update(world: "World", tasks: TaskSet, mission: "MissionConfig", upd: dict[str, Any]) -> dict[str, list[str]]:
    rev = int(upd["rev"])
    out: dict[str, list[str]] = {"changed": [], "cancelled": [], "added": []}
    new_zones = [[tuple(p) for p in z] for z in upd.get("nofly", [])]
    if new_zones:
        world.nofly.extend(new_zones)
        shapes = [Polygon(z) for z in new_zones]
        for t in tasks.unfinished():
            path = LineString(t.waypoints) if len(t.waypoints) >= 2 else None
            body = Polygon(t.polygon) if len(t.polygon) >= 3 else None
            if not any((body is not None and body.intersects(s)) or (path is not None and path.intersects(s))
                       for s in shapes):
                continue
            area = t.polygon if len(t.polygon) >= 3 else geo.rect(*geo.bbox(t.waypoints))
            wps = lawnmower(geo.rect(*geo.bbox(area)), area, mission.swath_width, world.nofly)
            if len(wps) < 2:
                t.status = TaskStatus.FAILED
                t.meta["cancelled"] = f"внутри бесполётной зоны (ревизия {rev})"
                out["cancelled"].append(t.id)
            else:
                t.waypoints, t.progress = wps, 0
                out["changed"].append(t.id)
    size = _cell_size(mission, tasks)
    for k, a in enumerate(upd.get("areas", [])):
        area = [tuple(p) for p in a]
        for r, c, cell in grid_cells(area, cell_size=size):
            poly = geo.clip_polygon(area, cell)
            if len(poly) < 3 or geo.area(poly) < mission.min_cell_area:
                continue
            wps = lawnmower(cell, area, mission.swath_width, world.nofly)
            if len(wps) < 2:
                continue
            tid = f"u{rev}{'' if k == 0 else f'_{k}'}_r{r:02d}_c{c:02d}"
            if tid in tasks:
                continue
            tasks.add(Task(id=tid, kind="sweep", waypoints=wps, polygon=poly,
                           meta={"row": r, "col": c, "area_m2": geo.area(poly), "rev": rev}))
            out["added"].append(tid)
    return out
