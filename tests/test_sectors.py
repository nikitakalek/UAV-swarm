"""M2-13: сектора, заданные пользователем, — формат, валидация, распределение по попаданию."""

import pytest
from pydantic import ValidationError

from swarmsim.missions.sectors import SECTORS_VERSION
from swarmsim.sim import run

HALF_S = [[0, 0], [1000, 0], [1000, 500], [0, 500]]
HALF_N = [[0, 500], [1000, 500], [1000, 1000], [0, 1000]]


def _sc(factory, sectors, **mission):
    return factory(agents={"count": 4}, allocator={"name": "manual_sectors"},
                   mission={"n_cells": 16, "sectors": sectors, **mission})


def test_format_is_versioned(scenario_factory):
    sc = _sc(scenario_factory, [{"id": "s", "polygon": HALF_S}, {"id": "n", "polygon": HALF_N}])
    assert sc.mission.sectors_version == SECTORS_VERSION
    with pytest.raises(ValidationError):
        _sc(scenario_factory, [{"id": "s", "polygon": HALF_S}, {"id": "n", "polygon": HALF_N}], sectors_version=99)


def test_rejects_overlap(scenario_factory):
    with pytest.raises(ValidationError, match="пересекаются"):
        _sc(scenario_factory, [{"id": "a", "polygon": [[0, 0], [1000, 0], [1000, 600], [0, 600]]},
                               {"id": "b", "polygon": HALF_N}])


def test_rejects_identical_sectors(scenario_factory):
    with pytest.raises(ValidationError, match="пересекаются"):
        _sc(scenario_factory, [{"id": "a", "polygon": HALF_S}, {"id": "b", "polygon": HALF_S},
                               {"id": "c", "polygon": HALF_N}])


def test_rejects_gap(scenario_factory):
    with pytest.raises(ValidationError, match="не покрывают"):
        _sc(scenario_factory, [{"id": "a", "polygon": [[0, 0], [1000, 0], [1000, 400], [0, 400]]},
                               {"id": "b", "polygon": HALF_N}])


def test_rejects_outside_area(scenario_factory):
    with pytest.raises(ValidationError, match="за границы"):
        _sc(scenario_factory, [{"id": "a", "polygon": [[0, -50], [1000, -50], [1000, 500], [0, 500]]},
                               {"id": "b", "polygon": HALF_N}])


def test_rejects_entering_nofly(scenario_factory):
    with pytest.raises(ValidationError, match="бесполётную"):
        scenario_factory(world={"nofly": [[[400, 400], [600, 400], [600, 600], [400, 600]]]},
                         agents={"count": 4}, allocator={"name": "manual_sectors"},
                         mission={"n_cells": 16, "sectors": [{"id": "s", "polygon": HALF_S},
                                                              {"id": "n", "polygon": HALF_N}]})


def test_sectors_may_leave_nofly_uncovered(scenario_factory):
    zone = [[400, 400], [600, 400], [600, 600], [400, 600]]
    s = [[0, 0], [1000, 0], [1000, 500], [600, 500], [600, 400], [400, 400], [400, 500], [0, 500]]
    n = [[0, 500], [400, 500], [400, 600], [600, 600], [600, 500], [1000, 500], [1000, 1000], [0, 1000]]
    sc = scenario_factory(world={"nofly": [zone]}, agents={"count": 4}, allocator={"name": "manual_sectors"},
                          mission={"n_cells": 16, "sectors": [{"id": "s", "polygon": s}, {"id": "n", "polygon": n}]})
    assert len(sc.mission.sectors) == 2


@pytest.mark.parametrize("method", ["centroid", "overlap"])
def test_tasks_assigned_by_membership(scenario_factory, method):
    sc = _sc(scenario_factory, [{"id": "s", "polygon": HALF_S, "agents": [0, 1]},
                                {"id": "n", "polygon": HALF_N, "agents": [2, 3]}], sector_assignment=method)
    res = run(sc, 1)
    for t in res.tasks:
        cy = sum(p[1] for p in t["polygon"]) / len(t["polygon"])
        assert t["sector"] == ("s" if cy < 500 else "n")
        assert t["done_by"] in ((0, 1) if t["sector"] == "s" else (2, 3))
    assert res.metrics["CR"] == 1.0


def test_clip_mode_keeps_tasks_inside_sectors(scenario_factory):
    diag_a = [[0, 0], [1000, 0], [0, 1000]]
    diag_b = [[1000, 0], [1000, 1000], [0, 1000]]
    sc = _sc(scenario_factory, [{"id": "a", "polygon": diag_a, "agents": [0]},
                                {"id": "b", "polygon": diag_b, "agents": [1]}], sector_assignment="clip")
    from swarmsim import geometry as geo

    res = run(sc, 1)
    for t in res.tasks:
        poly = diag_a if t["sector"] == "a" else diag_b
        for p in t["waypoints"]:
            # галс может выходить за наклонную границу не более чем на полполосы обзора
            inside = geo.point_in_polygon(tuple(p), poly)
            near = min(geo.point_segment_distance(tuple(p), poly[i], poly[(i + 1) % 3]) for i in range(3))
            assert inside or near <= sc.mission.swath_width / 2
    assert res.metrics["CR"] == 1.0


def test_failed_sector_agent_tasks_go_to_same_sector_first(scenario_factory):
    sc = _sc(scenario_factory, [{"id": "s", "polygon": HALF_S, "agents": [0, 1]},
                                {"id": "n", "polygon": HALF_N, "agents": [2, 3]}])
    sc = sc.model_copy(update={"events": []})
    from swarmsim.scenario import Scenario

    data = sc.model_dump(mode="json")
    data["events"] = [{"t": 40, "type": "agent_failure", "agent": 2}]
    res = run(Scenario.model_validate(data), 1)
    moved = [e for e in res.events.of_type("assign") if res.events.root_cause(e.seq).type == "agent_failure"]
    assert moved and all(e.data["to"] == 3 for e in moved)
    assert res.metrics["CR"] == 1.0
