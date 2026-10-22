"""M2-8: журнал событий и объяснений — причина каждого переназначения восстанавливается."""

import pytest

from swarmsim.eventlog import ROOT_TYPES, EventLog
from swarmsim.sim import run


def _res(scenario_factory):
    sc = scenario_factory(agents={"count": 6}, mission={"n_cells": 16},
                          events=[{"t": 60, "type": "agent_failure", "agent": 1},
                                  {"t": 90, "type": "agent_failure", "agent": 4},
                                  {"t": 120, "type": "new_target", "pos": [300, 700], "search_radius": 50},
                                  {"t": 130, "type": "comms_loss", "duration": 40}])
    return run(sc, 2)


def test_every_assignment_has_root_cause(scenario_factory):
    log = _res(scenario_factory).events
    changes = log.of_type("assign", "reassign", "unassign", "task_released")
    assert changes
    for e in changes:
        root = log.root_cause(e.seq)
        assert root.type in ROOT_TYPES, (e, root)
        assert e.data.get("reason")
    # назначения после отказа ведут к тому же отказу, что и освобождение задачи
    for e in log.of_type("assign"):
        root = log.root_cause(e.seq)
        if root.type == "agent_failure":
            released = [r for r in log.of_type("task_released")
                        if r.data["task"] == e.data["task"] and r.seq < e.seq]
            assert released and log.root_cause(released[-1].seq).seq == root.seq


def test_log_saved_with_run_and_roundtrips(tmp_path, scenario_factory):
    res = _res(scenario_factory)
    out = res.save(tmp_path / "run")
    log = EventLog.from_jsonl(out / "events.jsonl")
    assert log.rows() == res.events.rows()
    tid = log.of_type("task_released")[0].data["task"]
    assert any("agent_failure" in line for line in log.explain_task(tid))


def test_cause_must_reference_earlier_event():
    log = EventLog()
    log.append(type="mission_start", t=0)
    with pytest.raises(ValueError):
        log.append(type="x", t=1, cause=5)
