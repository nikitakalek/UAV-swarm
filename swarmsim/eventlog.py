"""Журнал событий и объяснений (M2-8). Формат — ``docs/m2/eventlog_format.md``.

Каждая запись: ``seq`` (порядковый номер), ``t`` (модельное время), ``type``,
``source`` (кто породил: ``scenario``, ``faults``, ``sim``, ``allocator:<имя>``,
``agent:<id>``, ``world``), ``cause`` (``seq`` события-причины) и ``data``.
По цепочке ``cause`` для любого переназначения восстанавливается исходная причина.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

FORMAT_VERSION = 1

# типы, которые могут быть корнем цепочки причин
ROOT_TYPES = {"mission_start", "agent_failure", "new_target", "comms_loss", "comms_restored"}


@dataclass
class Event:
    type: str
    t: float
    source: str = "sim"
    data: dict[str, Any] = field(default_factory=dict)
    cause: Optional[int] = None
    seq: int = -1

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


class EventLog:
    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, event: Event | None = None, /, **kw: Any) -> Event:
        """``append(Event(...))`` или ``append(type=..., t=..., ...)``. Возвращает запись с ``seq``."""
        ev = event if event is not None else Event(**kw)
        ev.seq = len(self._events)
        if ev.cause is not None and not (0 <= ev.cause < ev.seq):
            raise ValueError(f"cause={ev.cause} должен ссылаться на более раннее событие")
        self._events.append(ev)
        return ev

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, seq: int) -> Event:
        return self._events[seq]

    def of_type(self, *types: str) -> list[Event]:
        return [e for e in self._events if e.type in types]

    def chain(self, seq: int) -> list[Event]:
        """Цепочка причин от события к корню: ``[event, cause, cause_of_cause, ...]``."""
        out = []
        cur: Optional[int] = seq
        while cur is not None:
            ev = self._events[cur]
            out.append(ev)
            cur = ev.cause
        return out

    def root_cause(self, seq: int) -> Event:
        return self.chain(seq)[-1]

    def explain_task(self, task_id: str) -> list[str]:
        """Человекочитаемая история назначений задачи с причинами."""
        lines = []
        for e in self._events:
            if e.data.get("task") != task_id:
                continue
            root = self.root_cause(e.seq)
            why = e.data.get("reason", "")
            lines.append(
                f"t={e.t:.1f} {e.type}: {e.data.get('from')} → {e.data.get('to')}"
                f" ({why}; корень: {root.type}#{root.seq} t={root.t:.1f})"
            )
        return lines

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": "swarmcore-eventlog", "version": FORMAT_VERSION}) + "\n")
            for e in self._events:
                f.write(e.to_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EventLog":
        log = cls()
        with open(path, encoding="utf-8") as f:
            header = json.loads(f.readline())
            if header.get("format") != "swarmcore-eventlog":
                raise ValueError("не журнал событий SwarmCore")
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    log._events.append(Event(**d))
        return log

    def rows(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._events]
