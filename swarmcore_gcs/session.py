"""Сессия миссии: запуск узлов-процессов, цикл наземной станции, снимок состояния для пульта.

Узлы — те же процессы ``swarmcore_node node``, что на стенде; станция говорит с ними по
UDP. Пульт и 3D-вид получают снимок состояния по SSE (``snapshot``) и шлют команды.
"""

from __future__ import annotations

import json
import math
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from swarmcore_gcs.mission import Mission, make_update
from swarmcore_node.gcs import GroundStation
from swarmcore_node.link import UdpLink

COMMAND_TEXT = {"rtl": "все домой", "pause": "пауза", "resume": "продолжить"}


def describe(src: int, e: dict[str, Any]) -> Optional[str]:
    """Событие узла → строка для ленты оператора (``None`` — не показывать)."""
    d, typ = e.get("data", {}), e["type"]
    if typ == "plan_published":
        if d.get("cause") == "start":
            return f"Лидер {src} разделил участок между {len(d['members'])} аппаратами"
        if d.get("gone"):
            return (f"Лидер {src}: работу аппарата {', '.join(map(str, d['gone']))} забрали "
                    f"{', '.join(map(str, d['members']))} (план v{d['version']})")
        return f"Лидер {src}: новый план v{d['version']} — {d.get('cause')}"
    if typ == "plan_rejected":
        v = (d.get("violations") or ["?"])[0]
        return f"Лидер {src}: план отклонён валидатором — {v}"
    if typ == "peer_lost":
        return f"Нет связи с аппаратом {d['agent']} ({d['silence_s']} с без heartbeat)"
    if typ == "peer_back":
        return f"Аппарат {d['agent']} снова на связи"
    if typ == "takeoff":
        return f"Аппарат {src} взлетел на {d['altitude']:.0f} м"
    if typ == "landed":
        return f"Аппарат {src} сел на площадку"
    if typ == "rtl":
        return f"Аппарат {src} возвращается: {d['reason']}"
    if typ == "task_done":
        return f"Аппарат {src} выполнил ячейку {d['task'].replace('cell_', '')}"
    if typ == "vehicle_error":
        return f"Аппарат {src}: ошибка автопилота при {d['command']} — {d['error']}"
    if typ == "self_failure":
        return f"Аппарат {src}: отказ борта"
    if typ == "mode" and d.get("to") == "DEGRADED" and str(d.get("reason", "")).startswith("план отклонён"):
        return None  # об отклонении плана уже сказал лидер
    if typ == "mode" and d.get("to") in ("NO_GCS", "DEGRADED"):
        return f"Аппарат {src}: режим {d['to']} — {d.get('reason', '')}"
    if typ == "mission_updated":
        return (f"Группа приняла изменение миссии (ревизия {d['rev']}): +{d['added']} ячеек, "
                f"перестроено {d['changed']}, отменено {d['cancelled']}")
    if typ == "paused":
        return "Группа зависла на месте (пауза)"
    if typ == "resumed":
        return "Группа продолжает миссию"
    return None


class MissionSession:
    def __init__(self, runs_dir: str | Path = "runs/gcs", port_base: int = 47000, gcs_port: int = 46999,
                 python: str = sys.executable) -> None:
        self.runs_dir = Path(runs_dir)
        self.port_base, self.gcs_port, self.python = port_base, gcs_port, python
        self.lock = threading.RLock()
        self.state = "idle"  # idle | running | finished | stopped
        self.mission: Optional[Mission] = None
        self.gcs: Optional[GroundStation] = None
        self.link: Optional[UdpLink] = None
        self.procs: dict[int, subprocess.Popen] = {}
        self.run_dir: Optional[Path] = None
        self.epoch = 0.0
        self.scale = 1.0
        self.t = 0.0
        self.feed: list[dict[str, Any]] = []  # лента оператора: t, src, kind, text
        self._feed_n = 0
        self._shown: set[tuple] = set()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.replay_ready = False
        self.version = 0  # растёт при каждой новой миссии — страницы перечитывают геометрию

    # ------------------------------------------------------------------ жизненный цикл
    def launch(self, mission: Mission) -> dict[str, Any]:
        with self.lock:
            if self.state == "running":
                raise RuntimeError("Миссия уже идёт — сначала остановите её")
            self._cleanup()
            self.mission = mission
            self.scale = max(1.0, min(20.0, mission.params.get("time_scale", 1.0)))
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_dir = self.runs_dir / stamp
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "mission_draft.json").write_text(
                json.dumps({"draft": mission.draft, "params": mission.params}, ensure_ascii=False, indent=1))
            self.epoch = time.time()
            n = mission.n
            sc = mission.scenario
            common = ["--n", str(n), "--time-scale", str(self.scale), "--epoch", repr(self.epoch),
                      "--port-base", str(self.port_base), "--gcs", f"127.0.0.1:{self.gcs_port}",
                      "--out", str(self.run_dir), "--battery-wh", str(mission.params["battery_wh"]),
                      "--linger", "5"]
            for i in range(n):
                hx, hy = sc.agents.home_of(i)
                cmd = [self.python, "-m", "swarmcore_node", "node", "--id", str(i), "--home", f"{hx},{hy}",
                       "--speed", str(sc.agents.speed), "--altitude", str(mission.altitudes()[i]), *common]
                self.procs[i] = subprocess.Popen(cmd, stdout=open(self.run_dir / f"node_{i}.stdout", "w"),
                                                 stderr=subprocess.STDOUT)
            self.link = UdpLink(("0.0.0.0", self.gcs_port), [("127.0.0.1", self.port_base + i) for i in range(n)])
            self.gcs = GroundStation(self.link, sc, hb_period=0.5)
            self.feed, self._feed_n, self._shown = [], 0, set()
            self.replay_ready = False
            self.state = "running"
            self.version += 1
            self._note("Оператор запустил миссию: " f"{n} аппарата, {len(mission.geometry()['cells'])} ячеек", "cmd")
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gcs-loop", daemon=True)
            self._thread.start()
            return {"mission_id": self.gcs.mission_id, "run_dir": str(self.run_dir)}

    def _loop(self) -> None:
        t_done = None
        while not self._stop.is_set():
            with self.lock:
                self.t = (time.time() - self.epoch) * self.scale
                self.gcs.step(self.t)
                self._pull_feed()
                if self.gcs.mission_complete() or self._all_gone():
                    t_done = t_done or time.time()
            if t_done and time.time() - t_done > 2.0:
                self._finish("finished")
                return
            time.sleep(0.05)

    def _all_gone(self) -> bool:
        return bool(self.procs) and all(p.poll() is not None for p in self.procs.values())

    def stop(self) -> None:
        self._finish("stopped")

    def _finish(self, state: str) -> None:
        self._stop.set()
        with self.lock:
            if self.state != "running":
                return
            self.state = state
            self._note("Миссия завершена" if state == "finished" else "Оператор остановил миссию", "cmd")
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in self.procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        with self.lock:
            if self.gcs is not None:
                self.gcs.save(self.run_dir)
            if self.link is not None:
                self.link.close()
                self.link = None
        try:
            from swarmcore_node.replay import write

            write(self.run_dir)
            self.replay_ready = True
        except Exception as e:  # noqa: BLE001 — повтор не обязателен для работы пульта
            self._note(f"Повтор не собран: {e}", "warn")

    def _cleanup(self) -> None:
        for p in self.procs.values():
            if p.poll() is None:
                p.kill()
        self.procs = {}
        if self.link is not None:
            self.link.close()
            self.link = None

    # ------------------------------------------------------------------ команды
    def command(self, cmd: str) -> None:
        with self.lock:
            if self.state != "running":
                raise RuntimeError("Миссия не идёт")
            name, _, arg = cmd.partition(":")
            if name == "gcs_link":
                self.gcs.silent = arg == "off"
                self._note("Связь пульта с группой отключена (имитация)" if self.gcs.silent
                           else "Связь пульта с группой восстановлена", "cmd")
                return
            if name == "fail" and arg.isdigit() and int(arg) in self.procs:
                self.gcs.command(cmd)
                self.procs[int(arg)].send_signal(signal.SIGKILL)  # отказ борта: узел исчезает сразу
                self.gcs.events.append({"t": round(self.t, 3), "type": "command", "cmd": cmd})
                self._note(f"Имитация: отказ аппарата {arg}", "fail")
                return
            if name not in ("rtl", "pause", "resume", "drain"):
                raise ValueError(f"Неизвестная команда {cmd!r}")
            self.gcs.command(cmd)
            text = {"drain": f"Имитация: разряд батареи аппарата {arg}",
                    "rtl": f"Оператор: аппарат {arg} домой" if arg else "Оператор: все домой"}.get(name)
            self._note(text or f"Оператор: {COMMAND_TEXT[name]}", "cmd")

    def update_mission(self, body: dict[str, Any]) -> int:
        with self.lock:
            if self.state != "running":
                raise RuntimeError("Миссия не идёт — изменять нечего")
            upd = make_update(self.mission, body)
            rev = self.gcs.add_update({"nofly": upd["nofly"], "areas": upd["areas"]})
            self.mission.updates.append({**upd, "rev": rev})
            self.version += 1
            what = "новая бесполётная зона" if upd["nofly"] else "дополнительный участок"
            self._note(f"Оператор: {what} (ревизия {rev})", "cmd")
            return rev

    # ------------------------------------------------------------------ лента и снимок
    def _note(self, text: str, kind: str, src: int = -1) -> None:
        self.feed.append({"t": round(self.t, 1), "src": src, "kind": kind, "text": text})

    def _pull_feed(self) -> None:
        g = self.gcs
        while self._feed_n < len(g.feed):
            e = g.feed[self._feed_n]
            self._feed_n += 1
            key = (e["type"], e["data"].get("agent"), e["data"].get("task"), e["data"].get("version"))
            if e["type"] == "mission_updated":
                key = ("mission_updated", e["data"]["rev"])
            if e["type"] == "plan_rejected":  # пока лидер ищет допустимый план, повторы не показываем
                key = ("plan_rejected", e["src"], e["data"].get("cause"))
            if e["type"] in ("peer_lost", "peer_back", "paused", "resumed", "task_done", "mission_updated",
                             "plan_rejected"):
                if key in self._shown:  # одно событие видят несколько узлов — показываем один раз
                    continue
                self._shown.add(key)
            text = describe(e["src"], e)
            if text:
                kind = {"plan_published": "plan", "peer_lost": "fail", "self_failure": "fail", "plan_rejected": "warn",
                        "vehicle_error": "warn", "rtl": "warn", "task_done": "done"}.get(e["type"], "info")
                self.feed.append({"t": round(e["t"], 1), "src": e["src"], "kind": kind, "text": text})

    def snapshot(self, feed_from: int = 0) -> dict[str, Any]:
        with self.lock:
            out: dict[str, Any] = {"state": self.state, "version": self.version, "t": round(self.t, 2),
                                   "scale": self.scale, "replay": self.replay_ready,
                                   "feed": self.feed[feed_from:], "feed_n": len(self.feed)}
            if self.gcs is None or self.mission is None:
                return out
            g, m = self.gcs, self.mission
            done: set[str] = set()
            cancelled: set[str] = set()
            agents = []
            for i in range(m.n):
                st = g.status.get(i)
                age = self.t - g.last_seen.get(i, -math.inf)
                if st is None:
                    agents.append({"id": i, "link": "нет", "mode": "—", "altitude": m.altitudes()[i]})
                    continue
                done.update(st.get("done", []))
                cancelled.update(st.get("cancelled", []))
                lat, lon = m.frame.to_ll(*st["pos"])
                lost = age > 3.0 and st.get("mode") != "DONE"
                agents.append({
                    "id": i, "lat": round(lat, 7), "lon": round(lon, 7), "x": round(st["pos"][0], 2),
                    "y": round(st["pos"][1], 2), "alt": round(st.get("alt", 0.0), 1),
                    "mode": "ОТКАЗ" if lost else st.get("mode"), "status": st.get("status"),
                    "current": st.get("current"), "energy_wh": st.get("energy_used_wh", 0.0),
                    "battery_wh": st.get("battery_wh"), "leader": st.get("leader"), "paused": st.get("paused"),
                    "issue": st.get("issue", ""), "link_s": round(age, 1), "lost": lost,
                    "altitude": m.altitudes()[i]})
            owner = {}
            if g.plan:
                for a, ids in g.plan["plans"].items():
                    for tid in ids:
                        owner[tid] = int(a)
            n_tasks = max((st.get("n_tasks") or 0 for st in g.status.values()), default=0) - len(cancelled)
            out.update({"mission_id": g.mission_id, "agents": agents, "done": sorted(done), "owner": owner,
                        "cancelled": sorted(cancelled), "rev": len(g.updates),
                        "plan_version": g.plan["version"] if g.plan else 0, "n_tasks": n_tasks,
                        "gcs_link": not g.silent})
            return out

    def geometry(self) -> Optional[dict[str, Any]]:
        with self.lock:
            return None if self.mission is None else {**self.mission.geometry(), "version": self.version}
