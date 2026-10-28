"""Наземная станция (минимальная): одна задача на группу, статус, команды оператора.

Станция ничего не решает за группу: рассылает миссию (повторяет, пока каждый узел не
подтвердит её в heartbeat), шлёт свой heartbeat (по нему узлы понимают, есть ли связь
с оператором) и команды (``rtl``, ``fail:<id>`` — имитация отказа для демонстрации).
Все heartbeat узлов пишутся в журнал станции — из него строятся картинка и проверки.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

from swarmcore_node.link import PeerLink
from swarmcore_node.messages import GCS_ID, Msg
from swarmsim.scenario import Scenario


class GroundStation:
    def __init__(self, link: PeerLink, scenario: Scenario, mission_id: Optional[str] = None,
                 hb_period: float = 1.0, cmd_repeats: int = 3) -> None:
        self.link = link
        self.scenario = scenario
        self.mission_id = mission_id or uuid.uuid4().hex[:8]
        self.hb_period = hb_period
        self.cmd_repeats = cmd_repeats
        self.silent = False  # имитация потери связи с наземной станцией
        self.t = 0.0
        self.seq = 0
        self.status: dict[int, dict[str, Any]] = {}
        self.last_seen: dict[int, float] = {}
        self.acked: set[int] = set()
        self.records: list[dict[str, Any]] = []  # heartbeat узлов: (t, src, payload), без ленты событий
        self.events: list[dict[str, Any]] = []  # действия станции
        self._cmds: list[dict[str, Any]] = []  # активные команды: рассылаются, пока не истечёт cmd_ttl
        self.cmd_ttl = 3.0
        self._cmd_n = 0
        self.plan: Optional[dict[str, Any]] = None  # последний план лидера
        self.updates: list[dict[str, Any]] = []  # ревизии миссии (новые зоны и участки), в метрах
        self.feed: list[dict[str, Any]] = []  # события узлов без повторов: src, seq, t, type, data
        self._feed_seen: set[tuple[int, int]] = set()
        self._last_hb = -math.inf

    @property
    def n(self) -> int:
        return self.scenario.agents.count

    def command(self, cmd: str) -> None:
        """Команда группе. Повторяется в каждом heartbeat станции ``cmd_ttl`` секунд; узлы исполняют её один раз."""
        self._cmd_n += 1
        self._cmds.append({"id": f"{self.mission_id}-{self._cmd_n}", "cmd": cmd, "until": self.t + self.cmd_ttl})
        self.events.append({"t": round(self.t, 3), "type": "command", "cmd": cmd})

    def step(self, now: float) -> None:
        self.t = now
        for m in self.link.recv():
            if m.kind == "plan" and m.mission_id == self.mission_id:
                if self.plan is None or m.payload["version"] > self.plan["version"]:
                    self.plan = m.payload
                continue
            if m.kind == "hb" and 0 <= m.src < self.n:
                for e in m.payload.get("ev", []):
                    key = (m.src, e["seq"])
                    if key not in self._feed_seen:
                        self._feed_seen.add(key)
                        self.feed.append({"src": m.src, **e})
                self.status[m.src] = m.payload
                self.last_seen[m.src] = now
                self.records.append({"t": round(now, 3), "src": m.src,
                                     **{k: v for k, v in m.payload.items() if k != "ev"}})
                if m.mission_id == self.mission_id:
                    self.acked.add(m.src)
        if self.silent or now - self._last_hb < self.hb_period - 1e-9:
            return
        self._last_hb = now
        self._cmds = [c for c in self._cmds if c["until"] >= now]
        payload: dict[str, Any] = {"cmds": [{"id": c["id"], "cmd": c["cmd"]} for c in self._cmds]}
        if self.updates:  # ревизии повторяются в каждом heartbeat: узел применит недостающие
            payload.update({"mission_id": self.mission_id, "updates": self.updates})
        self.seq += 1
        self.link.send(Msg(kind="gcs_hb", src=GCS_ID, seq=self.seq, t=now, payload=payload))
        if len(self.acked) < self.n:
            self.seq += 1
            self.link.send(Msg(kind="mission", src=GCS_ID, seq=self.seq, t=now, mission_id=self.mission_id,
                               payload={"scenario": self.scenario.model_dump(mode="json")}))

    def mission_complete(self, stale: float = 5.0) -> bool:
        """Все узлы, которые ещё на связи, в ``DONE``, а все задачи выполнены."""
        live = [i for i in range(self.n) if self.t - self.last_seen.get(i, -math.inf) <= stale]
        if not live:
            return False
        return all(self.status[i].get("mode") == "DONE" for i in live) and self.tasks_left() == 0

    def add_update(self, upd: dict[str, Any]) -> int:
        """Новая ревизия миссии: ``{"nofly": [...], "areas": [...]}`` в метрах. Возвращает номер."""
        rev = len(self.updates) + 1
        self.updates.append({"rev": rev, "nofly": upd.get("nofly", []), "areas": upd.get("areas", [])})
        self.events.append({"t": round(self.t, 3), "type": "mission_update", "rev": rev})
        return rev

    def tasks_left(self) -> Optional[int]:
        """Сколько задач ещё не сделано и не отменено (по самому свежему составу задач)."""
        closed: set[str] = set()
        total = None
        for st in self.status.values():
            closed.update(st.get("done", []))
            closed.update(st.get("cancelled", []))
            if st.get("rev", 0) == len(self.updates) and st.get("n_tasks") is not None:
                total = max(total or 0, st["n_tasks"])
        return None if total is None else total - len(closed)

    def table(self) -> str:
        rows = [f"t={self.t:7.1f}  осталось задач: {self.tasks_left()}"]
        for i in range(self.n):
            st = self.status.get(i)
            age = self.t - self.last_seen.get(i, -math.inf)
            if st is None:
                rows.append(f"  #{i}: нет связи")
                continue
            lost = "  [НЕТ СВЯЗИ]" if age > 3 else ""
            rows.append(f"  #{i}: {st.get('mode', '?'):8} лидер={st.get('leader')} задача={st.get('current') or '-':14} "
                        f"сделано={len(st.get('done', []))}/{st.get('n_tasks', '?')} "
                        f"заряд={st.get('energy_used_wh', 0):.1f} Вт·ч  {st.get('issue', '')}{lost}")
        return "\n".join(rows)

    def save(self, outdir: str | Path) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "gcs_heartbeats.jsonl", "w") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / "gcs_events.json", "w") as f:
            json.dump({"mission_id": self.mission_id, "events": self.events, "updates": self.updates,
                       "scenario": self.scenario.model_dump(mode="json")}, f, ensure_ascii=False, indent=1)
        return out
