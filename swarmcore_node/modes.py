"""Конечный автомат режимов узла (М1).

Режим вычисляется по флагам с приоритетом ``DONE > RTL > DEGRADED > NO_GCS > NORMAL > IDLE``:

* ``IDLE`` — нет миссии;
* ``NORMAL`` — миссия, связь с группой и наземной станцией в порядке;
* ``NO_GCS`` — нет связи с наземной станцией: **миссия продолжается** (решения на борту);
* ``DEGRADED`` — потерян сосед и план ещё не пересчитан, или пересчёт отклонён валидатором;
* ``RTL`` — возврат на базу: команда оператора, низкий заряд, выполнено своё и миссия завершена;
* ``DONE`` — миссия завершена, аппарат на земле.

Аварийные режимы самого автопилота (failsafe PX4) этим автоматом не отключаются.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Mode(str, Enum):
    IDLE = "IDLE"
    NORMAL = "NORMAL"
    NO_GCS = "NO_GCS"
    DEGRADED = "DEGRADED"
    RTL = "RTL"
    DONE = "DONE"


@dataclass
class ModeInputs:
    has_mission: bool = False
    gcs_ok: bool = True
    peer_issue: Optional[str] = None  # причина деградации или None
    rtl_reason: Optional[str] = None
    done: bool = False


class ModeMachine:
    def __init__(self) -> None:
        self.mode = Mode.IDLE
        self.reason = ""

    @staticmethod
    def decide(x: ModeInputs) -> tuple[Mode, str]:
        if x.done:
            return Mode.DONE, "миссия завершена, аппарат на земле"
        if not x.has_mission:
            return Mode.IDLE, "нет миссии"
        if x.rtl_reason:
            return Mode.RTL, x.rtl_reason
        if x.peer_issue:
            return Mode.DEGRADED, x.peer_issue
        if not x.gcs_ok:
            return Mode.NO_GCS, "нет связи с наземной станцией — миссия продолжается"
        return Mode.NORMAL, ""

    def step(self, x: ModeInputs) -> Optional[tuple[Mode, Mode, str]]:
        """Обновляет режим. Возвращает ``(было, стало, причина)`` при смене режима, иначе ``None``."""
        new, why = self.decide(x)
        if new == self.mode:
            self.reason = why
            return None
        old, self.mode, self.reason = self.mode, new, why
        return old, new, why
