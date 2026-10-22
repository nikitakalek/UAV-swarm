"""Миссия поиска (M2-5): сетка ячеек + «змейка» внутри каждой ячейки → ``TaskSet``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from swarmsim import geometry as geo
from swarmsim.missions.base import MissionPlugin, Task, TaskSet
from swarmsim.planning.coverage import grid_cells, lawnmower

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.scenario import MissionConfig
    from swarmsim.world import World


class SearchMission(MissionPlugin):
    name = "search"

    def decompose(self, mission: "MissionConfig", world: "World") -> TaskSet:
        area = mission.area_polygon(world)
        ts = TaskSet()
        cells = grid_cells(area, n_cells=mission.n_cells, cell_size=mission.cell_size)
        if mission.sectors and mission.sector_assignment == "clip":
            from swarmsim.missions.sectors import clip_cells_to_sectors

            for task in clip_cells_to_sectors(cells, mission.sectors, mission.swath_width, world.nofly,
                                              mission.min_cell_area):
                task.priority = _prob_mass(world, task.polygon)
                ts.add(task)
            return ts
        for r, c, cell in cells:
            poly = geo.clip_polygon(area, cell)
            if len(poly) < 3 or geo.area(poly) < mission.min_cell_area:
                continue
            wps = lawnmower(cell, area, mission.swath_width, world.nofly)
            if len(wps) < 2:
                continue
            ts.add(Task(
                id=f"cell_r{r:02d}_c{c:02d}", kind="sweep", waypoints=wps, polygon=poly,
                priority=_prob_mass(world, poly), meta={"row": r, "col": c, "area_m2": geo.area(poly)},
            ))
        if mission.sectors:
            from swarmsim.missions.sectors import assign_tasks_to_sectors

            assign_tasks_to_sectors(ts, mission.sectors, method=mission.sector_assignment)
        return ts

    def on_new_target(self, mission, world, tasks, pos, radius, t):
        area = mission.area_polygon(world)
        x, y = pos
        cell = geo.rect(x - radius, y - radius, x + radius, y + radius)
        poly = geo.clip_polygon(area, cell)
        if len(poly) < 3:
            return []
        wps = lawnmower(cell, area, mission.swath_width, world.nofly)
        if len(wps) < 2:
            return []
        k = sum(1 for tk in tasks if tk.kind == "resweep")
        task = Task(id=f"extra_{k:02d}", kind="resweep", waypoints=wps, polygon=poly,
                    priority=2.0, t_created=t, meta={"reason": "new_target", "pos": [x, y]})
        tasks.add(task)
        return [task]


def _prob_mass(world: "World", poly) -> float:
    """Доля априорной вероятности цели в полигоне (вес задачи для U(M) и Pd-планирования)."""
    pm = world.prob_map
    if pm is None:
        return 1.0
    ny, nx = pm.p.shape
    x0, y0, x1, y1 = geo.bbox(poly)
    j0, j1 = max(0, int((x0 - pm.x0) // pm.cell)), min(nx - 1, int((x1 - pm.x0) // pm.cell))
    i0, i1 = max(0, int((y0 - pm.y0) // pm.cell)), min(ny - 1, int((y1 - pm.y0) // pm.cell))
    s = 0.0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            cx, cy = pm.x0 + (j + 0.5) * pm.cell, pm.y0 + (i + 0.5) * pm.cell
            if geo.point_in_polygon((cx, cy), poly):
                s += float(pm.p[i, j])
    return float(np.round(s, 6))
