"""Сквозной прогон: отдельные процессы-узлы по UDP на localhost + наземная станция + отказ."""

from __future__ import annotations

import json
import random
import subprocess
import sys

import pytest
import yaml

MISSION = {
    "name": "udp_test", "world": {"size": [100, 80], "targets": {"count": 0}},
    "agents": {"count": 3, "homes": [[0, 0], [4, 0], [8, 0]], "speed": 5, "altitude": 15, "altitude_step": 4},
    "sensor": {"swath_width": 20}, "mission": {"n_cells": 6},
}


@pytest.mark.slow
def test_demo_over_udp_with_failure(tmp_path):
    mission = tmp_path / "m.yaml"
    mission.write_text(yaml.safe_dump(MISSION, allow_unicode=True))
    base = random.randint(20000, 40000)
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, "-m", "swarmcore_node", "demo", "--mission", str(mission), "--fail", "2@25",
                        "--time-scale", "10", "--port-base", str(base), "--gcs-port", str(base - 1),
                        "--out", str(out), "--quiet", "--timeout", "600"],
                       capture_output=True, text=True, timeout=240)
    summary = json.loads((out / "summary.json").read_text())
    assert all(summary["checks"].values()), (summary, r.stdout[-2000:], r.stderr[-2000:])
    assert summary["realloc_s"]["2"] <= 5.0
    assert (out / "demo.png").exists()

    # 3D-повтор: экспорт журналов прогона
    from swarmcore_node.replay import write

    paths = write(out)
    data = json.loads(paths[0].read_text())
    assert data["meta"]["n"] == 3 and len(data["tasks"]) == data["meta"]["n_tasks"]
    assert data["fails"][0]["agent"] == 2
    assert all(len(tr["t"]) > 0 for tr in data["tracks"].values())
    assert any(e["kind"] == "plan" and "аппарата 2" in e["text"] for e in data["events"])
    assert "window.REPLAY_DATA" in paths[1].read_text()
