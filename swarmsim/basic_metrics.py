"""Минимальный сборщик метрик для стенда М2.

Полноценный ``MetricsRecorder`` (INT-1) — задача интеграции; этот класс даёт
базовый набор по определениям ``docs/03_experiment_program.md`` §4, чтобы
стенд с первого дня выдавал CSV и проверку детерминизма. Ключи с префиксом
``wall_`` — настенное время, они исключаются из сравнения прогонов.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from swarmsim.missions.base import TaskStatus

if TYPE_CHECKING:  # pragma: no cover
    from swarmsim.agents.base import Agent
    from swarmsim.missions.base import TaskSet
    from swarmsim.world import World

DETECT_LEVELS = (0.5, 0.9)


class BasicMetrics:
    def __init__(self, min_separation: float) -> None:
        self.min_sep = min_separation
        self.t_detect: dict[float, Optional[float]] = {x: None for x in DETECT_LEVELS}
        self.n_col = 0
        self.n_nm = 0
        self._col_pairs: set[tuple[int, int]] = set()
        self._nm_pairs: set[tuple[int, int]] = set()
        self.realloc_times: list[float] = []
        self.recoveries: list[dict[str, Any]] = []
        self.wall_alloc_s: list[float] = []
        self.wall_step_s = 0.0
        self.n_steps = 0
        self.t_last = 0.0
        self.gcs_down_s = 0.0
        self.done_during_gcs_loss = 0

    # ------------------------------------------------------------------ события цикла
    def on_failure_detected(self, agent_id: int, t_fail: float, t_detect: float, released: list[str]) -> None:
        self.recoveries.append({"agent": agent_id, "t_fail": t_fail, "t_detect": t_detect,
                                "tasks": list(released), "t_recovered": None if released else t_detect})

    def on_allocation(self, t: float, t_root: Optional[float], wall_s: float) -> None:
        self.wall_alloc_s.append(wall_s)
        if t_root is not None:
            self.realloc_times.append(t - t_root)

    def on_task_done(self, gcs_ok: bool) -> None:
        if not gcs_ok:
            self.done_during_gcs_loss += 1

    # ------------------------------------------------------------------ шаг
    def record(self, t: float, world: "World", agents: dict[int, "Agent"], tasks: "TaskSet", comms: Any,
               gcs_ok: bool = True, dt: float = 0.0) -> list[dict]:
        self.n_steps += 1
        self.t_last = t
        if not gcs_ok:
            self.gcs_down_s += dt
        act = world.active_targets()
        if act:
            frac = sum(tg.p_cum for tg in act) / len(act)
            for x in DETECT_LEVELS:
                if self.t_detect[x] is None and frac >= x - 1e-12:
                    self.t_detect[x] = t
        self._separation(agents)
        for rec in self.recoveries:
            if rec["t_recovered"] is None and all(
                tasks[tid].status in (TaskStatus.IN_PROGRESS, TaskStatus.DONE) for tid in rec["tasks"]
            ):
                rec["t_recovered"] = t
        return []

    def _separation(self, agents: dict[int, "Agent"]) -> None:
        live = [(i, a.state) for i, a in sorted(agents.items()) if a.alive and a.state.alt > 0]
        col, nm = set(), set()
        for k in range(len(live)):
            i, si = live[k]
            for m in range(k + 1, len(live)):
                j, sj = live[m]
                d = math.sqrt((si.x - sj.x) ** 2 + (si.y - sj.y) ** 2 + (si.alt - sj.alt) ** 2)
                if d < self.min_sep:
                    col.add((i, j))
                if d < 2 * self.min_sep:
                    nm.add((i, j))
        self.n_col += len(col - self._col_pairs)
        self.n_nm += len(nm - self._nm_pairs)
        self._col_pairs, self._nm_pairs = col, nm

    # ------------------------------------------------------------------ итог
    def finalize(self, world: "World", agents: dict[int, "Agent"], tasks: "TaskSet", comms: Any,
                 t_end: float) -> dict[str, Any]:
        n = len(tasks)
        done = [t for t in tasks if t.status == TaskStatus.DONE]
        unfinished = tasks.unfinished()
        any_alive = any(a.alive for a in agents.values())
        rec_times = [r["t_recovered"] - r["t_fail"] for r in self.recoveries if r["t_recovered"] is not None]
        n_dup = sum(max(0, t.done_count - 1) for t in tasks)
        n_lost = sum(1 for t in unfinished if t.assignee is None) if any_alive else 0
        targets = world.targets
        n_agents = len(agents)
        bytes_sent = getattr(comms, "bytes_sent", 0)
        T = max(self.t_last, 1e-9)
        m: dict[str, Any] = {
            "N": n_agents,
            "M_tasks": n,
            "M_targets": len(targets),
            "CR": len(done) / n if n else math.nan,
            "tasks_done": len(done),
            "tasks_unfinished": len(unfinished),
            "T_detect_50": _nan(self.t_detect[0.5]),
            "T_detect_90": _nan(self.t_detect[0.9]),
            "Pd_mean": sum(tg.p_cum for tg in targets) / len(targets) if targets else math.nan,
            "detected_fraction": (sum(tg.detected_t is not None for tg in targets) / len(targets)
                                  if targets else math.nan),
            "makespan": max((t.t_done for t in done), default=math.nan),
            "t_sim_end": self.t_last,
            "E_total_Wh": sum(a.energy_used_wh for a in agents.values()),
            "distance_total_m": sum(a.distance_m for a in agents.values()),
            "n_failures": len(self.recoveries),
            "n_reallocations": len(self.realloc_times),
            "T_realloc_mean": _mean(self.realloc_times),
            "T_realloc_max": max(self.realloc_times, default=math.nan),
            "T_recovery_mean": _mean(rec_times),
            "T_recovery_max": max(rec_times, default=math.nan),
            "n_unrecovered": sum(r["t_recovered"] is None for r in self.recoveries),
            "N_msg": getattr(comms, "n_sent", 0),
            "B_per_agent_Bps": bytes_sent / (n_agents * T) if n_agents else 0.0,
            "gcs_down_s": self.gcs_down_s,
            "A_gcs_tasks": self.done_during_gcs_loss,
            "N_col": self.n_col,
            "N_nm": self.n_nm,
            "N_dup": n_dup,
            "N_lost": n_lost,
            "wall_alloc_ms_mean": 1000 * _mean(self.wall_alloc_s),
            "wall_step_ms_mean": 1000 * self.wall_step_s / max(self.n_steps, 1),
        }
        m["valid"] = bool(m["N_col"] == 0 and m["N_dup"] == 0 and m["N_lost"] == 0)
        return m


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _nan(x: Optional[float]) -> float:
    return math.nan if x is None else x
