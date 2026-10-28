"""Бортовой узел swarm_node: группа узлов, которые общаются только через канал (М1).

Критерии MVP (docs/05): отказ → перераспределение ≤ 5 с, ни одна задача не потеряна и не
продублирована; работа без наземной станции; недопустимый план не исполняется.
"""

from __future__ import annotations

import pytest

from swarmcore_node.messages import Msg, PlanPayload
from swarmcore_node.modes import Mode, ModeInputs, ModeMachine
from swarmcore_node.node import NodeConfig
from swarmcore_node.testing import LocalSwarm

SMALL = {
    "name": "small", "world": {"size": [160, 120], "targets": {"count": 0}},
    "agents": {"count": 3, "homes": [[0, 0], [4, 0], [8, 0]], "speed": 5, "altitude": 15, "altitude_step": 4},
    "sensor": {"swath_width": 20}, "mission": {"n_cells": 9},
}


def _ok(s):
    assert s["lost"] == 0, s
    assert s["duplicates"] == 0, s
    assert s["done"] == s["n_tasks"] > 0


def test_nominal_mission_completes_and_everyone_lands():
    sw = LocalSwarm(SMALL).run(900)
    s = sw.summary()
    _ok(s)
    assert set(s["modes"].values()) == {"DONE"}
    assert s["leaders"] == [0]
    for n in sw.nodes:
        assert n.log.of_type("landed"), f"узел {n.id} не сел"


@pytest.mark.parametrize("victim", [1, 2])
def test_follower_failure_is_repaired_within_5s(victim):
    sw = LocalSwarm(SMALL)
    sw.at(40, lambda: sw.kill(victim))
    s = sw.run(900).summary()
    _ok(s)
    assert s["realloc_s"][victim] is not None and s["realloc_s"][victim] <= 5.0


def test_leader_failure_next_node_takes_over():
    sw = LocalSwarm(SMALL)
    sw.at(30, lambda: sw.kill(0))
    s = sw.run(900).summary()
    _ok(s)
    assert s["leaders"] == [0, 1]
    assert s["realloc_s"][0] <= 5.0


def test_unfinished_part_of_task_continues_from_progress():
    """Сирота продолжается с непройденной точки галса, а не с начала."""
    sw = LocalSwarm(SMALL)
    progress = {}

    def fail():
        n = sw.nodes[1]
        cur = n.agent.current
        progress["task"], progress["k"] = cur, n.tasks[cur].progress
        sw.kill(1)

    sw.at(45, fail)
    sw.run(900)
    assert progress["k"] > 0
    survivor = sw.nodes[0]
    adopted = [e for e in survivor.log.of_type("assign") if e.data["task"] == progress["task"] and e.data["from"] == 1]
    assert adopted, "задача отказавшего переназначена"
    _ok(sw.summary())


def test_message_loss_30_percent():
    sw = LocalSwarm(SMALL, loss=0.3, seed=7)
    sw.at(50, lambda: sw.kill(2))
    _ok(sw.run(900).summary())


def test_mission_continues_without_gcs():
    sw = LocalSwarm(SMALL)
    sw.at(20, lambda: setattr(sw.gcs, "silent", True))
    s = sw.run(900).summary()
    _ok(s)
    modes = [e.data["to"] for e in sw.nodes[0].log.of_type("mode")]
    assert "NO_GCS" in modes and modes[-1] == "DONE"


def test_operator_rtl_brings_everyone_home():
    sw = LocalSwarm(SMALL)
    sw.at(30, lambda: sw.gcs.command("rtl"))
    sw.run(900)
    for n in sw.nodes:
        assert n.rtl_reason == "команда оператора"
        assert n.mode == Mode.DONE and n.log.of_type("landed")


def test_invalid_plan_is_not_executed():
    sw = LocalSwarm(SMALL, NodeConfig(battery_wh=1.0)).run(30)
    leader = sw.nodes[0]
    assert leader.log.of_type("plan_rejected")
    assert leader.mode == Mode.DEGRADED
    assert not any(n.log.of_type("takeoff") for n in sw.nodes), "никто не взлетел по недопустимому плану"


def test_low_battery_rtl_hands_tasks_over():
    sw = LocalSwarm(SMALL)

    def drain():  # имитация разряда (реальный расход больше расчётного)
        sw.nodes[1].energy_used_wh = sw.nodes[1].cfg.battery_wh * 0.78

    sw.at(40, drain)
    s = sw.run(900).summary()
    _ok(s)
    assert "низкий заряд" in (sw.nodes[1].rtl_reason or "")
    causes = [e.data["cause"] for e in sw.nodes[0].log.of_type("plan_published")]
    assert any("выбыли:[1]" in c for c in causes)


def test_ten_agents_two_failures():
    sc = dict(SMALL, world={"size": [1000, 1000], "targets": {"count": 0}},
              agents={"count": 10, "speed": 10, "altitude": 30, "altitude_step": 4}, sensor={"swath_width": 40},
              mission={"n_cells": 25})
    sw = LocalSwarm(sc, NodeConfig(battery_wh=400))
    sw.at(150, lambda: sw.kill(3))
    sw.at(240, lambda: sw.kill(0))
    _ok(sw.run(3000).summary())


def test_altitude_layers_are_separated():
    sw = LocalSwarm(SMALL)
    sw.run(5)
    alts = sorted(n.motion.state().alt for n in sw.nodes)
    assert all(b - a >= 3.0 for a, b in zip(alts, alts[1:]))


def test_every_reassignment_has_a_reason_and_root_cause():
    sw = LocalSwarm(SMALL)
    sw.at(40, lambda: sw.kill(1))
    sw.run(900)
    log = sw.nodes[0].log
    moved = [e for e in log.of_type("assign") if e.data["from"] == 1]
    assert moved
    for e in moved:
        assert e.data["reason"]
        assert log.root_cause(e.seq).type == "peer_lost"


# ------------------------------------------------------------------ протокол и режимы
def test_plan_payload_roundtrip_and_ordering():
    p = PlanPayload(version=3, leader=1, members=[1, 2], excluded=[0], plans={1: ["a"], 2: ["b"]})
    m = Msg.decode(Msg(kind="plan", src=1, payload=p.to_dict()).encode())
    q = PlanPayload.from_dict(m.payload)
    assert q == p
    assert PlanPayload(3, 0, [], [], {}).key() > PlanPayload(3, 1, [], [], {}).key() > PlanPayload(2, 0, [], [], {}).key()
    assert Msg.decode(b"garbage") is None


@pytest.mark.parametrize("inputs,mode", [
    (ModeInputs(), Mode.IDLE),
    (ModeInputs(has_mission=True), Mode.NORMAL),
    (ModeInputs(has_mission=True, gcs_ok=False), Mode.NO_GCS),
    (ModeInputs(has_mission=True, gcs_ok=False, peer_issue="x"), Mode.DEGRADED),
    (ModeInputs(has_mission=True, peer_issue="x", rtl_reason="y"), Mode.RTL),
    (ModeInputs(has_mission=True, rtl_reason="y", done=True), Mode.DONE),
])
def test_mode_priority(inputs, mode):
    assert ModeMachine.decide(inputs)[0] == mode


def test_mode_machine_reports_transitions_once():
    mm = ModeMachine()
    assert mm.step(ModeInputs(has_mission=True)) == (Mode.IDLE, Mode.NORMAL, "")
    assert mm.step(ModeInputs(has_mission=True)) is None
    assert mm.step(ModeInputs(has_mission=True, gcs_ok=False))[1] == Mode.NO_GCS
