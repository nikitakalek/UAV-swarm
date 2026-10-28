"""Экспорт прогона в ``replay.json`` для 3D-повтора (``swarmcore_node/replay3d.html``).

Источник — папка прогона (``demo``/``gcs``): журнал heartbeat станции, журналы узлов,
сценарий. Одинаково работает для кинематической модели и для PX4 SITL.

    python -m swarmcore_node replay runs/demo            # → runs/demo/replay.json и replay.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from swarmcore_node.demo import _load, analyze
from swarmcore_node.mission_update import apply_update
from swarmsim.missions import decompose
from swarmsim.scenario import Scenario
from swarmsim.world import World

TEMPLATE = Path(__file__).with_name("replay3d.html")


def _r(v: float, k: int = 2) -> float:
    return round(float(v), k)


def export(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    meta, hbs, events = _load(run)
    summary = analyze(run) if not (run / "summary.json").exists() else json.loads((run / "summary.json").read_text())
    sc = Scenario.model_validate(meta["scenario"])
    world = World.from_config(sc.world, np.random.default_rng(0))
    tasks = decompose(sc.mission, world)
    for upd in meta.get("updates", []):  # изменения миссии в полёте
        apply_update(world, tasks, sc.mission, upd)
    n = sc.agents.count

    # --- треки аппаратов (по heartbeat, которые видела станция)
    tracks: dict[str, dict[str, list]] = {str(i): {"t": [], "x": [], "y": [], "z": [], "mode": [], "cur": [], "st": []}
                                          for i in range(n)}
    first_done: dict[str, float] = {}
    for h in hbs:
        tr = tracks.get(str(h["src"]))
        if tr is None:
            continue
        tr["t"].append(_r(h["t"]))
        tr["x"].append(_r(h["pos"][0]))
        tr["y"].append(_r(h["pos"][1]))
        tr["z"].append(_r(h.get("alt", 0.0)))
        tr["mode"].append(h.get("mode", ""))
        tr["cur"].append(h.get("current") or "")
        tr["st"].append(h.get("status", ""))
        for tid in h.get("done", []):
            first_done.setdefault(tid, _r(h["t"]))

    # --- назначения: начальные и переданные после событий
    owner0: dict[str, int] = {}
    moves: list[dict[str, Any]] = []
    done_by: dict[str, int] = {}
    timeline: list[dict[str, Any]] = []
    seen_lost: set[int] = set()
    for i, evs in sorted(events.items()):
        for e in evs:
            d, t, typ = e["data"], _r(e["t"]), e["type"]
            if typ == "assign":
                if d["from"] is None:
                    owner0.setdefault(d["task"], d["to"])
                else:
                    moves.append({"t": t, "task": d["task"], "from": d["from"], "to": d["to"], "why": d["reason"]})
            elif typ == "task_done":
                done_by.setdefault(d["task"], i)
            elif typ == "plan_published":
                gone = d.get("gone") or []
                if d.get("cause") == "start":
                    text = f"Лидер {i} разделил участок между {len(d['members'])} аппаратами (план v{d['version']})"
                else:
                    text = (f"Лидер {i}: новый план v{d['version']} — работу аппарата {', '.join(map(str, gone))} "
                            f"забрали {', '.join(map(str, d['members']))}" if gone else
                            f"Лидер {i}: новый план v{d['version']} ({d.get('cause')})")
                timeline.append({"t": t, "kind": "plan", "agent": i, "text": text})
            elif typ == "peer_lost" and d["agent"] not in seen_lost:
                seen_lost.add(d["agent"])
                timeline.append({"t": t, "kind": "lost", "agent": d["agent"],
                                 "text": f"Аппарат {i} заметил потерю связи с аппаратом {d['agent']} "
                                         f"({d['silence_s']} с без heartbeat)"})
            elif typ == "takeoff":
                timeline.append({"t": t, "kind": "takeoff", "agent": i,
                                 "text": f"Аппарат {i} взлетел на {d['altitude']:.0f} м"})
            elif typ == "landed":
                timeline.append({"t": t, "kind": "landed", "agent": i, "text": f"Аппарат {i} сел на базе"})
            elif typ == "rtl":
                timeline.append({"t": t, "kind": "rtl", "agent": i, "text": f"Аппарат {i} возвращается: {d['reason']}"})
            elif typ == "mode" and d["to"] in ("NO_GCS", "DEGRADED"):
                timeline.append({"t": t, "kind": "mode", "agent": i, "text": f"Аппарат {i}: режим {d['to']}"})
    # журнал отказавшего узла не сохраняется (процесс убит) — взлёт восстанавливаем по телеметрии
    for i in range(n):
        if i in events:
            continue
        tr = tracks[str(i)]
        k = next((k for k, z in enumerate(tr["z"]) if z > 1.0), None)
        if k is not None:
            timeline.append({"t": tr["t"][k], "kind": "takeoff", "agent": i,
                             "text": f"Аппарат {i} взлетел на {sc.agents.altitude + i * sc.agents.altitude_step:.0f} м"})
    fails = []
    for e in meta["events"]:
        if e.get("cmd", "").startswith("fail:"):
            a = int(e["cmd"].split(":")[1])
            fails.append({"t": _r(e["t"]), "agent": a})
            timeline.append({"t": _r(e["t"]), "kind": "fail", "agent": a, "text": f"Отказ аппарата {a}"})
        elif e.get("type") == "gcs_silent":
            timeline.append({"t": _r(e["t"]), "kind": "mode", "agent": -1, "text": "Связь с наземной станцией потеряна"})

    moves.sort(key=lambda m: m["t"])
    out_tasks = []
    for t in tasks:
        own = owner0.get(t.id)
        mv = [m for m in moves if m["task"] == t.id]
        td = first_done.get(t.id)
        by = done_by.get(t.id)
        if by is None and td is not None:  # журнал отказавшего узла не сохранился — берём владельца на тот момент
            by = next((m["to"] for m in reversed(mv) if m["t"] <= td), own)
        out_tasks.append({"id": t.id, "poly": [[_r(x), _r(y)] for x, y in t.polygon],
                          "wps": [[_r(x), _r(y)] for x, y in t.waypoints], "owner0": own, "t_done": td, "done_by": by})
        if td is not None:
            timeline.append({"t": td, "kind": "done", "agent": by if by is not None else -1,
                             "text": f"Ячейка {t.id.replace('cell_', '')} выполнена"
                                     + (f" аппаратом {by}" if by is not None else "")})
    timeline.sort(key=lambda e: e["t"])

    t0 = min((tr["t"][0] for tr in tracks.values() if tr["t"]), default=0.0)
    return {
        "version": 1,
        "meta": {"name": sc.name, "description": sc.description, "mission_id": meta["mission_id"], "n": n,
                 "speed": sc.agents.speed, "swath": sc.mission.swath_width, "t0": t0, "t_end": summary["t_end"],
                 "checks": summary["checks"], "realloc_s": summary["realloc_s"],
                 "min_vertical_sep_m": summary["min_vertical_sep_m"], "n_tasks": summary["n_tasks"]},
        "world": {"bounds": [[_r(x), _r(y)] for x, y in world.bounds],
                  "nofly": [[[_r(x), _r(y)] for x, y in z] for z in world.nofly],
                  "homes": [list(sc.agents.home_of(i)) for i in range(n)],
                  "altitudes": [sc.agents.altitude + i * sc.agents.altitude_step for i in range(n)]},
        "tasks": out_tasks, "moves": moves, "fails": fails, "events": timeline, "tracks": tracks,
    }


def write(run_dir: str | Path, html: bool = True) -> list[Path]:
    run = Path(run_dir)
    data = export(run)
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out = [run / "replay.json"]
    out[0].write_text(js, encoding="utf-8")
    if html:
        page = TEMPLATE.read_text(encoding="utf-8")
        doc = ("<!doctype html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\">"
               "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
               f"<script>window.REPLAY_DATA = {js};</script></head><body>\n{page}\n</body></html>\n")
        (run / "replay.html").write_text(doc, encoding="utf-8")
        out.append(run / "replay.html")
    return out
