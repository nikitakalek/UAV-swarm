"""Сообщения между узлами и наземной станцией. Формат — JSON в одной UDP-датаграмме.

| kind | от кого | что внутри |
|---|---|---|
| ``mission`` | НС | сценарий миссии (``swarmsim.scenario.Scenario`` как dict) и его ``mission_id`` |
| ``gcs_hb`` | НС | признак жизни наземной станции, команда (``rtl``, ``fail:<id>``) |
| ``hb`` | узел | позиция, режим, текущая задача, выполненные задачи, прогресс, энергия, версия плана |
| ``plan`` | лидер | версия, состав группы, план каждого аппарата, причины назначений |

Все операции над задачами идемпотентны: «задача выполнена» и «прогресс задачи» только
растут, поэтому повтор, потеря или переупорядочивание сообщений не ломают состояние.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

PROTOCOL_VERSION = 1
GCS_ID = -1


@dataclass
class Msg:
    kind: str
    src: int
    seq: int = 0
    t: float = 0.0  # часы отправителя (только для журнала; тайм-ауты — по часам получателя)
    mission_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    v: int = PROTOCOL_VERSION

    def encode(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode()

    @classmethod
    def decode(cls, data: bytes) -> Optional["Msg"]:
        try:
            d = json.loads(data.decode())
            if d.get("v") != PROTOCOL_VERSION:
                return None
            return cls(**d)
        except (ValueError, TypeError, UnicodeDecodeError):
            return None


@dataclass
class PlanPayload:
    """Содержимое сообщения ``plan``. Сравнение версий — по ``(version, -leader)``."""

    version: int
    leader: int
    members: list[int]  # кто участвует в плане (живые и готовые аппараты)
    excluded: list[int]  # кого лидер считает выбывшими
    plans: dict[int, list[str]]
    reasons: dict[str, str] = field(default_factory=dict)
    cause: str = ""  # почему пересчитали: start | peer_lost:2 | new_tasks | peer_back:2 ...

    def key(self) -> tuple[int, int]:
        return (self.version, -self.leader)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["plans"] = {str(k): v for k, v in self.plans.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanPayload":
        d = dict(d)
        d["plans"] = {int(k): list(v) for k, v in d["plans"].items()}
        return cls(**d)
