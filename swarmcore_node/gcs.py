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
        self.records: list[dict[str, Any]] = []  # heartbeat узлов: (t, src, payload)
        self.events: list[dict[str, Any]] = []  # действия станции
        self._cmds: list[list[Any]] = []
        self._last_hb = -math.inf

    @property
    def n(self) -> int:
        return self.scenario.agents.count

    def command(self, cmd: str) -> None:
        self._cmds.append([cmd, self.cmd_repeats])
        self.events.append({"t": round(self.t, 3), "type": "command", "cmd": cmd})

    def step(self, now: float) -> None:
        self.t = now
        for m in self.link.recv():
            if m.kind == "hb" and 0 <= m.src < self.n:
                self.status[m.src] = m.payload
                self.last_seen[m.src] = now
                self.records.append({"t": round(now, 3), "src": m.src, **m.payload})
                if m.mission_id == self.mission_id:
                    self.acked.add(m.src)
        if self.silent or now - self._last_hb < self.hb_period - 1e-9:
            return
        self._last_hb = now
        payload: dict[str, Any] = {}
        if self._cmds:
            payload["cmd"] = self._cmds[0][0]
            self._cmds[0][1] -= 1
            if self._cmds[0][1] <= 0:
                self._cmds.pop(0)
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

    def tasks_left(self) -> Optional[int]:
        done: set[str] = set()
        total = None
        for st in self.status.values():
            done.update(st.get("done", []))
            total = st.get("n_tasks", total)
        return None if total is None else total - len(done)

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
            json.dump({"mission_id": self.mission_id, "events": self.events,
                       "scenario": self.scenario.model_dump(mode="json")}, f, ensure_ascii=False, indent=1)
        return out
