"""M2-4 / M2-7 / M2-9: детерминизм цикла, manifest, события, неидеальный канал."""

import json
import random
import time

import pytest

from swarmsim.sim import Simulation, run
from swarmsim.stubs import RoundRobinStubAllocator


def test_same_seed_identical_metrics_and_log(scenario_factory):
    sc = scenario_factory(events=[{"t": 100, "type": "agent_failure", "agent": 1}])
    a, b = run(sc, 5), run(sc, 5)
    assert a.deterministic_metrics() == b.deterministic_metrics()
    assert a.metrics_hash() == b.metrics_hash()
    assert a.events_hash() == b.events_hash()
    assert a.trajectory == b.trajectory


def test_different_seed_different_world(scenario_factory):
    sc = scenario_factory()
    assert run(sc, 1).world["targets"] != run(sc, 2).world["targets"]


def test_independent_of_wall_clock(scenario_factory, monkeypatch):
    """should_trigger и цикл не зависят от настенного времени: подменяем часы на случайные."""
    sc = scenario_factory(events=[{"t": 60, "type": "agent_failure", "agent": 0},
                                  {"t": 90, "type": "comms_loss", "duration": 30}])
    ref = run(sc, 3)
    rnd = random.Random(0)
    monkeypatch.setattr(time, "perf_counter", lambda: rnd.random() * 1e6)
    monkeypatch.setattr(time, "time", lambda: rnd.random() * 1e9)
    other = Simulation(sc, 3, clock=lambda: rnd.random() * 1e6).run()
    assert ref.metrics_hash() == other.metrics_hash()
    assert ref.events_hash() == other.events_hash()


def test_manifest_written_with_config_hash(tmp_path, scenario_factory):
    sc = scenario_factory()
    res = run(sc, 1)
    out = res.save(tmp_path / "r")
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["config_hash"] == sc.config_hash()
    assert man["seed"] == 1
    assert man["metrics_hash"] == res.metrics_hash()
    for f in man["files"]:
        assert (out / f).exists()
    # хэш конфига чувствителен к изменению сценария
    assert scenario_factory(agents={"count": 5}).config_hash() != sc.config_hash()


def test_nominal_mission_completes_and_valid(scenario_factory):
    res = run(scenario_factory(), 1)
    m = res.metrics
    assert m["CR"] == 1.0
    assert m["valid"]
    assert m["N_dup"] == 0 and m["N_lost"] == 0 and m["N_col"] == 0
    assert m["T_detect_50"] <= m["makespan"]


def test_failure_releases_and_reallocates(scenario_factory):
    res = run(scenario_factory(events=[{"t": 50, "type": "agent_failure", "agent": 2}],
                               faults={"detect_delay": 1.0}), 1)
    log = res.events
    assert log.of_type("agent_failure")[0].t == 50
    det = log.of_type("failure_detected")[0]
    assert det.t == pytest.approx(51.0)
    released = log.of_type("task_released")
    assert released and all(e.cause == det.seq for e in released)
    assert res.metrics["CR"] == 1.0
    assert res.metrics["n_failures"] == 1 and res.metrics["n_unrecovered"] == 0
    # отказавший агент ничего не завершал после отказа
    assert all(e.source != "agent:2" or e.t <= 50 for e in log.of_type("task_done"))


def test_random_failures_reproducible(scenario_factory):
    sc = scenario_factory(agents={"count": 10}, mission={"n_cells": 16},
                          faults={"random_failures": {"fraction": 0.2, "t_window": [10, 200]}})
    a, b = run(sc, 11), run(sc, 11)
    fa = [(e.t, e.data["agent"]) for e in a.events.of_type("agent_failure")]
    assert len(fa) == 2
    assert fa == [(e.t, e.data["agent"]) for e in b.events.of_type("agent_failure")]
    c = run(sc, 12)
    assert fa != [(e.t, e.data["agent"]) for e in c.events.of_type("agent_failure")]


def test_new_target_creates_task_and_it_gets_done(scenario_factory):
    res = run(scenario_factory(events=[{"t": 30, "type": "new_target", "pos": [500, 500], "search_radius": 60}]), 1)
    tasks = {t["id"]: t for t in res.tasks}
    assert "extra_00" in tasks and tasks["extra_00"]["status"] == "done"
    assert res.metrics["M_targets"] == 11


def test_no_reallocation_while_gcs_down(scenario_factory):
    sc = scenario_factory(events=[{"t": 40, "type": "comms_loss", "duration": 100, "scope": "gcs"},
                                  {"t": 50, "type": "agent_failure", "agent": 1}])
    res = run(sc, 1)
    allocs = [e for e in res.events.of_type("allocation") if e.data["trigger"] != "mission_start"]
    assert allocs and allocs[0].t >= 140.0
    assert res.metrics["gcs_down_s"] == pytest.approx(100.0)
    assert res.metrics["CR"] == 1.0


class _Spy:
    """Распределитель-обёртка, записывающий порядок доставленных сообщений."""

    name = "spy"

    def __init__(self):
        self.inner = RoundRobinStubAllocator()
        self.seen = []

    def allocate(self, state, tasks):
        return self.inner.allocate(state, tasks)

    def should_trigger(self, t, tasks, pending):
        return self.inner.should_trigger(t, tasks, pending)

    def on_message(self, m):
        self.seen.append((m.t_deliver, m.src, m.seq))


def test_lossy_channel_deterministic_and_ordered(scenario_factory):
    """M2-9: задержки, дрожание и потери — те же результаты при том же зерне; порядок доставки детерминирован."""
    sc = scenario_factory(comms={"model": "stub_lossy", "delay": 0.3, "jitter": 0.4, "loss": 0.3},
                          events=[{"t": 60, "type": "agent_failure", "agent": 3},
                                  {"t": 80, "type": "comms_loss", "duration": 20, "scope": "all"}])
    s1, s2 = _Spy(), _Spy()
    r1 = Simulation(sc, 9, allocator=s1).run()
    r2 = Simulation(sc, 9, allocator=s2).run()
    assert s1.seen == s2.seen and len(s1.seen) > 100
    assert s1.seen == sorted(s1.seen)
    assert r1.metrics_hash() == r2.metrics_hash()
    assert r1.events_hash() == r2.events_hash()
    assert r1.metrics["N_msg"] > len(s1.seen)  # часть сообщений потеряна
    s3 = _Spy()
    Simulation(sc, 10, allocator=s3).run()
    assert s3.seen != s1.seen


def test_paired_design_world_independent_of_allocator(scenario_factory):
    """Одно зерно → тот же мир при любом распределителе (парный дизайн docs/03)."""

    class Reverse(RoundRobinStubAllocator):
        name = "reverse"

        def allocate(self, state, tasks):
            a = super().allocate(state, tasks)
            a.plans = {k: list(reversed(v)) for k, v in a.plans.items()}
            return a

    sc = scenario_factory()
    r1 = run(sc, 4)
    r2 = Simulation(sc, 4, allocator=Reverse()).run()
    assert [(t["x"], t["y"]) for t in r1.world["targets"]] == [(t["x"], t["y"]) for t in r2.world["targets"]]


def test_external_allocator_by_module_path(scenario_factory):
    sc = scenario_factory(allocator={"name": "swarmsim.stubs:RoundRobinStubAllocator"})
    assert run(sc, 1).metrics["CR"] == 1.0
