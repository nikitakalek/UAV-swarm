"""SwarmNode — бортовой узел SwarmCore: один процесс на аппарат (М1).

Узел знает только свой аппарат (``MotionSource``) и соседей через канал (``PeerLink``).

* **Состояние миссии** — копия на каждом узле: сценарий, задачи, план, кто что сделал.
  Выполненные задачи и прогресс передаются в каждом heartbeat и только растут, поэтому
  копии сходятся при потерях и повторах сообщений.
* **Распределение** делает лидер — узел с минимальным id среди живых. Если лидер выбыл,
  его роль сразу берёт следующий: копия состояния у него уже есть.
* **Отказ соседа** — нет heartbeat дольше ``peer_timeout``. Лидер пересчитывает план
  (``SectorAllocator``: сироты — тому, кто закончит раньше), проверяет его валидатором и
  рассылает. Недопустимый план не публикуется и не исполняется.
* **Без наземной станции** миссия продолжается (режим ``NO_GCS``).
* **Журнал**: каждое решение с причиной (``EventLog``), трасса полёта.

Время — аргумент ``step(now)``: в тестах модельное, на стенде — реальное.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from swarmcore_node.link import PeerLink
from swarmcore_node.messages import GCS_ID, Msg, PlanPayload
from swarmcore_node.modes import Mode, ModeInputs, ModeMachine
from swarmcore_node.vehicle import make_commands
from swarmsim.agents.base import Agent, MotionSource
from swarmsim.eventlog import EventLog
from swarmsim.interfaces import AgentView, SwarmSnapshot
from swarmsim.missions import TaskSet, decompose
from swarmsim.planning.validator import PlanValidator
from swarmsim.registry import make_allocator
from swarmsim.scenario import Scenario
from swarmsim.world import World


@dataclass
class NodeConfig:
    heartbeat_period: float = 0.5  # с
    peer_timeout: float = 2.5  # нет heartbeat дольше — сосед выбыл
    gcs_timeout: float = 5.0
    replan_min_interval: float = 1.0  # не чаще — защита от «дребезга»
    battery_wh: float = 200.0
    power_w: float = 250.0
    reserve: float = 0.2
    min_sep: float = 5.0
    min_sep_vertical: float = 3.0
    z_range: tuple[float, float] = (0.0, 120.0)
    allocator: str = "sector"
    allocator_params: dict[str, Any] = field(default_factory=dict)
    # inline — команды аппарату мгновенные (кинематика); thread — в фоне (MAVSDK: взлёт до 90 с)
    vehicle_commands: str = "inline"


@dataclass
class PeerInfo:
    id: int
    last_seen: float
    pos: Optional[tuple[float, float]] = None
    home: Optional[tuple[float, float]] = None
    mode: str = Mode.IDLE.value
    current: Optional[str] = None
    energy_used_wh: float = 0.0
    plan_version: int = 0


class SwarmNode:
    def __init__(self, node_id: int, motion: MotionSource, link: PeerLink,
                 config: Optional[NodeConfig] = None) -> None:
        self.id = node_id
        self.motion = motion
        self.link = link
        self.cfg = config or NodeConfig()
        self.log = EventLog()
        self.trace: list[dict[str, Any]] = []
        self.modes = ModeMachine()
        self.validator = PlanValidator(battery_wh=self.cfg.battery_wh, power_w=self.cfg.power_w,
                                       reserve=self.cfg.reserve, min_sep=self.cfg.min_sep,
                                       min_sep_vertical=self.cfg.min_sep_vertical, z_range=self.cfg.z_range)

        self.t: Optional[float] = None
        self.seq = 0
        self.crashed = False

        self.mission: Optional[Scenario] = None
        self.mission_id = ""
        self.world: Optional[World] = None
        self.tasks: Optional[TaskSet] = None
        self.agent: Optional[Agent] = None
        self.allocator: Any = None
        self.home: tuple[float, float] = (0.0, 0.0)
        self.altitude = 0.0
        self.altitudes: dict[int, float] = {}
        self.speed = 1.0

        self.peers: dict[int, PeerInfo] = {}
        self.dead: set[int] = set()
        self.departed: set[int] = set()  # завершили миссию и замолчали — не отказ
        self.plan: Optional[PlanPayload] = None
        self.max_version = 0
        self.plan_issue: Optional[str] = None
        self.rtl_reason: Optional[str] = None
        self.phase = "ground"  # ground → takeoff → air → landing → ground
        self.vehicle = make_commands(self.cfg.vehicle_commands, f"vehicle-{node_id}")
        self._phase_t = 0.0
        self.finished = False
        self.energy_used_wh = 0.0
        self.gcs_seen = -math.inf
        self._last_hb = -math.inf
        self._last_replan = -math.inf
        self._last_rejected = ""
        self._done_logged: set[str] = set()
        self._root: Optional[int] = None  # событие-причина для следующего пересчёта

    # ================================================================== цикл
    def step(self, now: float) -> None:
        if self.crashed:
            return
        dt = 0.0 if self.t is None else max(0.0, now - self.t)
        self.t = now
        for m in self.link.recv():
            if m.src != self.id:
                self._on_msg(m)
        self._poll_vehicle()
        if self.mission is not None and not self.finished:
            self._detect_failures()
            if self.is_leader():
                self._maybe_replan()
            self._execute(dt)
        self._update_mode()
        if now - self._last_hb >= self.cfg.heartbeat_period - 1e-9:
            self._last_hb = now
            self._send_heartbeat()
        self._record_trace()

    def crash(self) -> None:
        """Имитация отказа борта: аппарат садится, узел замолкает."""
        if self.crashed:
            return
        self._event("self_failure", data={"reason": "отказ борта (имитация)"})
        self.crashed = True
        self.vehicle.run_now(self.motion.kill)  # не ждём: узел замолкает сразу

    # ================================================================== роли
    def alive_ids(self) -> list[int]:
        return sorted([self.id] + [j for j in self.peers if j not in self.dead and j not in self.departed])

    def is_leader(self) -> bool:
        return self.alive_ids()[0] == self.id

    def leader(self) -> int:
        return self.alive_ids()[0]

    def usable_ids(self) -> list[int]:
        """Кто может брать задачи: живые, с известной позицией, не на возврате."""
        out = []
        if self.rtl_reason is None:
            out.append(self.id)
        for j, p in self.peers.items():
            if j in self.dead or j in self.departed or p.pos is None or p.mode in (Mode.RTL.value, Mode.DONE.value):
                continue
            out.append(j)
        return sorted(out)

    # ================================================================== сообщения
    def _send(self, kind: str, payload: dict[str, Any]) -> None:
        self.seq += 1
        self.link.send(Msg(kind=kind, src=self.id, seq=self.seq, t=self.t or 0.0,
                           mission_id=self.mission_id, payload=payload))

    def _on_msg(self, m: Msg) -> None:
        if m.kind == "mission":
            self._on_mission(m)
            return
        if m.kind == "gcs_hb":
            self.gcs_seen = self.t
            self._on_command(m.payload.get("cmd"))
            if m.mission_id and m.mission_id != self.mission_id and "scenario" in m.payload:
                self._on_mission(m)
            return
        if self.mission is None or m.mission_id != self.mission_id:
            return
        if m.kind == "hb":
            self._on_heartbeat(m)
        elif m.kind == "plan":
            self._adopt(PlanPayload.from_dict(m.payload), source=m.src)

    def _on_mission(self, m: Msg) -> None:
        if m.mission_id == self.mission_id:
            return
        if self.phase != "ground":
            self._event("mission_ignored", data={"mission_id": m.mission_id, "reason": "аппарат в полёте"})
            return
        sc = Scenario.model_validate(m.payload["scenario"])
        if not (0 <= self.id < sc.agents.count):
            self._event("mission_ignored", data={"mission_id": m.mission_id, "reason": "id вне состава группы"})
            return
        self.mission, self.mission_id = sc, m.mission_id
        self.world = World.from_config(sc.world.model_copy(update={"targets": sc.world.targets.model_copy(
            update={"count": 0, "positions": []})}), np.random.default_rng(0))
        self.tasks = decompose(sc.mission, self.world)
        self.home = tuple(self.motion.state().pos)
        self.altitudes = {i: sc.agents.altitude + i * sc.agents.altitude_step for i in range(sc.agents.count)}
        self.altitude = self.altitudes[self.id]
        self.speed = sc.agents.speed
        self.agent = Agent(self.id, self.motion, self.home)
        self.allocator = make_allocator(self.cfg.allocator, **self.cfg.allocator_params)
        self.peers = {j: PeerInfo(j, last_seen=self.t) for j in range(sc.agents.count) if j != self.id}
        self.dead.clear()
        self.departed.clear()
        self.plan, self.finished, self.rtl_reason = None, False, None
        ev = self._event("mission_start", data={"mission_id": m.mission_id, "tasks": len(self.tasks),
                                                "agents": sc.agents.count, "altitude": self.altitude})
        self._root = ev.seq

    def _on_command(self, cmd: Optional[str]) -> None:
        if not cmd:
            return
        if cmd == "rtl" and self.rtl_reason is None and self.mission is not None:
            self._start_rtl("команда оператора")
        elif cmd.startswith("fail:") and int(cmd.split(":")[1]) == self.id:
            self.crash()

    def _on_heartbeat(self, m: Msg) -> None:
        p = self.peers.get(m.src)
        if p is None:
            return
        pl = m.payload
        p.last_seen = self.t
        p.pos, p.home = tuple(pl["pos"]), tuple(pl["home"])
        p.mode, p.current = pl["mode"], pl.get("current")
        p.energy_used_wh, p.plan_version = pl.get("energy_used_wh", 0.0), pl.get("plan_version", 0)
        self.departed.discard(m.src)
        if m.src in self.dead:
            self.dead.discard(m.src)
            ev = self._event("peer_back", data={"agent": m.src})
            self._root = ev.seq
        self._merge_progress(m.src, pl.get("done", []), pl.get("progress", {}))
        # лидер досылает план тем, у кого он устарел (сообщение plan могло потеряться)
        if self.plan and self.plan.leader == self.id and p.plan_version < self.plan.version:
            self._send("plan", self.plan.to_dict())

    def _merge_progress(self, src: int, done: list[str], progress: dict[str, int]) -> None:
        for tid in done:
            if tid in self.tasks and not self.tasks[tid].is_finished:
                self.tasks[tid].mark_done(src, self.t)
        for tid, k in progress.items():
            if tid in self.tasks and (self.agent.current != tid):
                task = self.tasks[tid]
                task.progress = max(task.progress, int(k))
        a = self.agent
        if a.current is not None and self.tasks[a.current].is_finished:
            a.set_plan(list(a.queue), self.tasks)  # задачу уже закрыл другой — бросаем её

    # ================================================================== отказы и план
    def _detect_failures(self) -> None:
        for j, p in sorted(self.peers.items()):
            if j in self.dead or j in self.departed or self.t - p.last_seen <= self.cfg.peer_timeout:
                continue
            if p.mode == Mode.DONE.value:
                self.departed.add(j)
                self._event("peer_departed", data={"agent": j})
            else:
                self.dead.add(j)
                ev = self._event("peer_lost", data={"agent": j, "silence_s": round(self.t - p.last_seen, 2)})
                self._root = ev.seq

    def _snapshot(self, members: list[int]) -> SwarmSnapshot:
        views: dict[int, AgentView] = {}
        open_ids = {t.id for t in self.tasks.unfinished()}
        for j in sorted({self.id, *self.peers}):
            if j == self.id:
                pos, home, cur, used = tuple(self.motion.state().pos), self.home, self.agent.current, self.energy_used_wh
            else:
                p = self.peers[j]
                pos, home, cur, used = p.pos or (0.0, 0.0), p.home or (0.0, 0.0), p.current, p.energy_used_wh
            cur = cur if cur in open_ids else None
            queue = [t for t in (self.plan.plans.get(j, []) if self.plan else []) if t in open_ids and t != cur]
            views[j] = AgentView(agent_id=j, pos=pos, home=home, alive=j in members, speed=self.speed,
                                 current_task=cur, queue=queue, energy_used_wh=used)
        return SwarmSnapshot(t=self.t, agents=views, world=self.world, gcs_reachable=self._gcs_ok())

    def _replan_cause(self, members: list[int]) -> Optional[str]:
        if self.plan is None:
            waiting = [j for j, p in self.peers.items() if j not in self.dead and p.pos is None]
            return None if waiting else "start"
        if set(members) != set(self.plan.members):
            open_ids = {t.id for t in self.tasks.unfinished()}
            # ушёл тот, у кого ещё есть работа (закончившие свою часть пересчёта не требуют)
            self._gone = sorted(j for j in set(self.plan.members) - set(members)
                                if open_ids & set(self.plan.plans.get(j, [])))
            new = sorted(set(members) - set(self.plan.members))
            if self._gone or new:
                return ";".join([f"выбыли:{self._gone}"] * bool(self._gone) + [f"вернулись:{new}"] * bool(new))
        planned = {t for ids in self.plan.plans.values() for t in ids}
        if any(t.id not in planned for t in self.tasks.unfinished()):
            return "новые задачи"
        return None

    def _maybe_replan(self) -> None:
        if self.t - self._last_replan < self.cfg.replan_min_interval:
            return
        members = self.usable_ids()
        self._gone: list[int] = []
        cause = self._replan_cause(members)
        if cause is None or not members:
            return
        self._last_replan = self.t
        snap = self._snapshot(members)
        asg = self.allocator.allocate(snap, self.tasks)
        plans = {j: asg.plans.get(j, []) for j in members}
        asg.plans = plans
        report = self.validator.validate(asg, snap, self.tasks, {j: self.altitudes[j] for j in members})
        if not report.ok:
            text = "; ".join(str(v) for v in report.violations[:5])
            self.plan_issue = f"план отклонён валидатором: {text}"
            if text != self._last_rejected:
                self._last_rejected = text
                self._event("plan_rejected", cause=self._root,
                            data={"cause": cause, "violations": [str(v) for v in report.violations]})
            return
        self._last_rejected = ""
        plan = PlanPayload(version=self.max_version + 1, leader=self.id, members=members,
                           excluded=sorted(self.dead), plans=plans, reasons=dict(asg.reasons), cause=cause)
        old = self.plan
        ev = self._event("plan_published", cause=self._root, data={
            "version": plan.version, "cause": cause, "gone": self._gone, "members": members, "excluded": plan.excluded,
            "load_m": asg.meta.get("load_m", {}), "mode": asg.meta.get("mode")})
        self._log_assignments(old, plan, ev.seq)
        self._root = None
        self._send("plan", plan.to_dict())
        self._adopt(plan, source=self.id)

    def _log_assignments(self, old: Optional[PlanPayload], new: PlanPayload, cause: int) -> None:
        before = {t: a for a, ids in (old.plans.items() if old else []) for t in ids}
        for a, ids in sorted(new.plans.items()):
            for t in ids:
                if before.get(t) != a:
                    self._event("assign", cause=cause, data={"task": t, "from": before.get(t), "to": a,
                                                             "reason": new.reasons.get(t, "")})

    def _adopt(self, plan: PlanPayload, source: int) -> None:
        if self.plan is not None and plan.key() <= self.plan.key():
            return
        self.plan = plan
        self.max_version = max(self.max_version, plan.version)
        if set(self.dead) <= set(plan.excluded) | set(plan.members):
            self.plan_issue = None
        if source != self.id:
            self._event("plan_adopted", data={"version": plan.version, "leader": plan.leader, "cause": plan.cause})
        mine = plan.plans.get(self.id, []) if self.id in plan.members and self.rtl_reason is None else []
        self.agent.set_plan(list(mine), self.tasks)  # взлёт — в _execute, когда в плане есть работа

    # ================================================================== исполнение
    @property
    def flying(self) -> bool:
        return self.phase == "air"

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._phase_t = self.t or 0.0

    def _takeoff(self) -> None:
        alt = self.altitude

        def go() -> None:
            self.motion.arm()
            self.motion.takeoff(alt)

        self._set_phase("takeoff")
        self._event("takeoff_started", data={"altitude": alt})
        self.vehicle.submit("takeoff", go)
        self._poll_vehicle()

    def _land(self) -> None:
        self._set_phase("landing")
        self.vehicle.submit("land", self.motion.land)
        self._poll_vehicle()

    def _poll_vehicle(self) -> None:
        """Результаты команд аппарату, выполненных в фоне."""
        for name, err in self.vehicle.poll():
            took = round((self.t or 0.0) - self._phase_t, 2)
            if err is not None:
                self._event("vehicle_error", data={"command": name, "error": str(err)})
                self._set_phase("ground" if name == "takeoff" else "air")
                if self.rtl_reason is None and self.mission is not None:
                    self._start_rtl(f"ошибка аппарата при {name}: {err}")
            elif name == "takeoff":
                self._set_phase("air")
                self.agent.status = "idle"
                self._event("takeoff", data={"altitude": self.altitude, "took_s": took})
            elif name == "land":
                self._set_phase("ground")
                self._event("landed", data={"energy_used_wh": round(self.energy_used_wh, 1), "took_s": took})

    def _start_rtl(self, reason: str) -> None:
        self.rtl_reason = reason
        self._event("rtl", data={"reason": reason})
        if self.agent is not None and self.tasks is not None:
            self.agent.set_plan([], self.tasks)
            if self.phase == "air" and self.agent.status in ("idle", "home"):
                self.agent.status = "transit"  # Agent сам уведёт аппарат домой

    def _execute(self, dt: float) -> None:
        a = self.agent
        has_work = a.current is not None or any(not self.tasks[t].is_finished for t in a.queue)
        if self.phase == "ground" and has_work and self.rtl_reason is None:
            self._takeoff()
        if self.phase in ("takeoff", "air", "landing"):
            self.energy_used_wh += self.cfg.power_w * dt / 3600.0
        if self.phase == "air":
            a.step(dt, self.tasks, self.world)
            self._check_battery()
            if a.current is None and not a.queue and a.status in ("home", "idle") and \
                    math.dist(a.state.pos, self.home) < 1.0:
                self._land()
        for t in self.tasks:
            if t.done_by == self.id and t.id not in self._done_logged:
                self._done_logged.add(t.id)
                self._event("task_done", data={"task": t.id})
        if self.phase == "ground" and (self.rtl_reason or not self.tasks.unfinished()):
            self.finished = True

    def _check_battery(self) -> None:
        if self.rtl_reason is not None:
            return
        cap = self.cfg.battery_wh
        to_home = math.dist(self.motion.state().pos, self.home) / max(self.speed, 1e-9)
        need = self.cfg.power_w * to_home / 3600.0
        if cap - self.energy_used_wh - need < self.cfg.reserve * cap:
            self._start_rtl("низкий заряд: остаётся резерв на возврат")

    # ================================================================== режим, телеметрия, журнал
    def _gcs_ok(self) -> bool:
        return self.t is not None and self.t - self.gcs_seen <= self.cfg.gcs_timeout

    def _update_mode(self) -> None:
        issue = self.plan_issue
        if self.plan is not None and not (self.dead <= set(self.plan.excluded)):
            issue = issue or f"потеряны аппараты {sorted(self.dead - set(self.plan.excluded))}, план пересчитывается"
        ch = self.modes.step(ModeInputs(has_mission=self.mission is not None, gcs_ok=self._gcs_ok(),
                                        peer_issue=issue, rtl_reason=self.rtl_reason, done=self.finished))
        if ch:
            self._event("mode", data={"from": ch[0].value, "to": ch[1].value, "reason": ch[2]})

    @property
    def mode(self) -> Mode:
        return self.modes.mode

    def _send_heartbeat(self) -> None:
        st = self.motion.state()
        payload: dict[str, Any] = {"pos": [st.x, st.y], "alt": st.alt, "home": list(self.home),
                                   "mode": self.mode.value, "leader": self.leader()}
        if self.mission is not None:
            a = self.agent
            payload.update({
                "current": a.current, "queue": list(a.queue), "flying": self.flying, "phase": self.phase,
                "status": a.status if self.phase == "air" else self.phase,
                "progress": {a.current: self.tasks[a.current].progress} if a.current else {},
                "done": sorted(t.id for t in self.tasks if t.is_finished),
                "n_tasks": len(self.tasks), "energy_used_wh": round(self.energy_used_wh, 3),
                "plan_version": self.plan.version if self.plan else 0,
                "dead": sorted(self.dead), "issue": self.plan_issue or self.rtl_reason or "",
            })
        self._send("hb", payload)

    def _event(self, type: str, cause: Optional[int] = None, data: Optional[dict] = None):
        return self.log.append(type=type, t=round(self.t or 0.0, 3), source=f"node:{self.id}", cause=cause,
                               data=data or {})

    def _record_trace(self) -> None:
        st = self.motion.state()
        if self.trace and self.t - self.trace[-1]["t"] < 0.5:
            return
        self.trace.append({"t": round(self.t, 2), "x": round(st.x, 2), "y": round(st.y, 2), "alt": round(st.alt, 2),
                           "mode": self.mode.value, "current": self.agent.current if self.agent else None,
                           "leader": int(self.leader() == self.id)})

    def save(self, outdir: str | Path) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        self.log.to_jsonl(out / f"node_{self.id}_events.jsonl")
        with open(out / f"node_{self.id}_trace.csv", "w", newline="") as f:
            if self.trace:
                w = csv.DictWriter(f, fieldnames=list(self.trace[0]))
                w.writeheader()
                w.writerows(self.trace)
        return out


__all__ = ["SwarmNode", "NodeConfig", "GCS_ID"]
