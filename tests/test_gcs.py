"""Наземная станция: координаты, черновик миссии с карты, предпросмотр, API и полный цикл миссии."""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request

import pytest

from swarmcore_gcs.geo import LocalFrame
from swarmcore_gcs.mission import DraftError, build, make_update, preview

F = LocalFrame(55.3768, 37.73)


def ll(x, y):
    return list(F.to_ll(x, y))


def draft(**params):
    return {"area": [ll(20, 20), ll(220, 20), ll(220, 170), ll(20, 170)],
            "nofly": [[ll(110, 80), ll(140, 80), ll(140, 105), ll(110, 105)]],
            "homes": [ll(0, 0), ll(6, 0), ll(12, 0)], "params": {"cell_size": 50, **params}}


def test_local_frame_roundtrip_and_scale():
    x, y = F.to_xy(*F.to_ll(123.4, -56.7))
    assert (round(x, 6), round(y, 6)) == (123.4, -56.7)
    # 0,001° широты ≈ 111 м
    assert abs(F.to_xy(F.lat0 + 0.001, F.lon0)[1] - 111.3) < 0.5


def test_build_and_preview():
    m = build(draft())
    assert m.n == 3 and m.scenario.mission.cell_size == 50
    assert m.altitudes() == {0: 15.0, 1: 19.0, 2: 23.0}
    p = preview(m)
    assert p["ok"], p["violations"]
    assert p["n_tasks"] == 12 and abs(p["area_m2"] - 30000) < 50
    assert {c["owner"] for c in p["cells"]} == {0, 1, 2}
    assert all(a["path"] and a["time_s"] > 0 for a in p["agents"])


@pytest.mark.parametrize("bad,msg", [
    ({"area": []}, "Участок"),
    ({"homes": []}, "Площадки"),
    ({"params": {"altitude_step": 2}}, "Шаг эшелонов"),
    ({"area": [ll(0, 0), ll(100, 100), ll(100, 0), ll(0, 100)]}, "пересекает сам себя"),
])
def test_bad_drafts_explain_the_problem(bad, msg):
    with pytest.raises(DraftError, match=msg):
        build({**draft(), **bad})


def test_preview_reports_validator_problems():
    p = preview(build(draft(battery_wh=10)))
    assert not p["ok"] and any("energy" in v for v in p["violations"])


def test_mission_update_checks():
    m = build(draft())
    u = make_update(m, {"nofly": [ll(60, 60), ll(90, 60), ll(90, 90), ll(60, 90)]})
    assert len(u["nofly"]) == 1 and not u["areas"]
    with pytest.raises(DraftError, match="выходит за район"):
        make_update(m, {"area": [ll(900, 0), ll(1000, 0), ll(1000, 100)]})
    with pytest.raises(DraftError):
        make_update(m, {})
    m.updates.append({**make_update(m, {"area": [ll(230, 40), ll(300, 40), ll(300, 140), ll(230, 140)]}), "rev": 1})
    g = m.geometry()
    assert g["rev"] == 1 and any(c["id"].startswith("u1_") for c in g["cells"]) and g["areas_extra"]


# ------------------------------------------------------------------ сервер: полный цикл по HTTP
@pytest.fixture
def server(tmp_path):
    from swarmcore_gcs.server import make_server
    from swarmcore_gcs.session import MissionSession

    base = random.randint(30000, 45000)
    srv = make_server("127.0.0.1", base + 50, MissionSession(tmp_path / "runs", port_base=base, gcs_port=base - 1))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{base + 50}", srv
    srv.RequestHandlerClass.session.stop()
    srv.shutdown()
    srv.server_close()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return {"status": e.code, **json.loads(e.read())}


def _state(url):
    return json.loads(urllib.request.urlopen(url + "/api/state", timeout=10).read())


@pytest.mark.slow
def test_full_mission_over_http(server):
    url, _ = server
    page = urllib.request.urlopen(url + "/").read().decode()
    assert "Пульт SwarmCore" in page and "3D-вид SwarmCore" in urllib.request.urlopen(url + "/view").read().decode()
    assert _post(url + "/api/command", {"cmd": "rtl"})["status"] == 400  # миссии ещё нет
    assert "mission_id" in _post(url + "/api/launch", draft(time_scale=20))
    assert _post(url + "/api/launch", draft())["status"] == 400  # вторая миссия поверх идущей
    time.sleep(2.5)
    assert _post(url + "/api/command", {"cmd": "fail:1"}) == {"ok": True}
    assert _post(url + "/api/mission/update", {"nofly": [ll(60, 120), ll(100, 120), ll(100, 160), ll(60, 160)]})["rev"] == 1
    assert _post(url + "/api/command", {"cmd": "boom"})["status"] == 400
    t0 = time.time()
    while time.time() - t0 < 120 and _state(url)["state"] == "running":
        time.sleep(1)
    s = _state(url)
    assert s["state"] == "finished"
    assert len(s["done"]) == s["n_tasks"] > 0
    texts = [e["text"] for e in s["feed"]]
    assert any("отказ аппарата 1" in t for t in texts)
    assert any("работу аппарата 1 забрали" in t for t in texts)
    assert any("изменение миссии (ревизия 1)" in t for t in texts)
    time.sleep(1)
    replay = json.loads(urllib.request.urlopen(url + "/replay.json").read())
    assert replay["meta"]["checks"]["нет дублей"] and replay["fails"][0]["agent"] == 1
