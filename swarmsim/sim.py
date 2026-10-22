"""SimLoop (M2-4, M2-7, M2-9): детерминированный цикл симуляции ``run(scenario, seed)``.

Порядок шага (docs/02, §4):
  0) события сценария (отказ, новая цель, потеря связи) по модельному времени;
  1) детекция отказов, освобождение задач отказавших;
  2) доставка сообщений — в детерминированном порядке ``(t_deliver, src, seq)``;
  3) перераспределение — по событию (``should_trigger``), если доступна наземная станция;
  4) шаг агентов (следование плану) и обнаружение целей;
  5) шаг мира, метрики, запись траекторий.

Детерминизм: фиксированный ``dt``, время шага ``t = k·dt`` (целый счётчик, без
накопления ошибки), независимые потоки случайных чисел из ``SeedSequence(seed)``,
никакой зависимости от настенного времени, кроме метрик ``wall_*``.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import swarmsim
from swarmsim import registry
from swarmsim.agents.base import Agent, MotionSource
from swarmsim.agents.kinematics import KinematicMotion
from swarmsim.basic_metrics import BasicMetrics
from swarmsim.eventlog import Event, EventLog
from swarmsim.interfaces import GCS_ID, AgentView, Assignment, Message, SwarmSnapshot
from swarmsim.missions import MISSIONS
from swarmsim.missions.base import TaskSet, TaskStatus
from swarmsim.scenario import AgentFailureEvent, AgentsConfig, CommsLossEvent, NewTargetEvent, Scenario
from swarmsim.world import World

MotionFactory = Callable[[int, tuple[float, float], float, AgentsConfig], MotionSource]

RNG_STREAMS = ("world", "sensing", "comms", "faults", "alloc")


def _accepts(fn: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def default_motion(i: int, home: tuple[float, float], alt: float, cfg: AgentsConfig) -> MotionSource:
    return KinematicMotion(i, home, v_cruise=cfg.speed, v_max=cfg.v_max, altitude=alt)


# =========================================================================== результат
@dataclass
class RunResult:
    scenario: Scenario
    seed: int
    allocator: str
    metrics: dict[str, Any]
    trajectory: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    events: EventLog
    world: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)

    def deterministic_metrics(self) -> dict[str, Any]:
        return {k: v for k, v in self.metrics.items() if not k.startswith("wall_")}

    def metrics_hash(self) -> str:
        return _hash_obj(self.deterministic_metrics())

    def events_hash(self) -> str:
        return _hash_obj(self.events.rows())

    def save(self, outdir: str | Path, parquet: bool = False) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        row = {"scenario": self.scenario.name, "allocator": self.allocator, "seed": self.seed, **self.metrics}
        _write_csv(out / "metrics.csv", [row])
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(row), f, ensure_ascii=False, indent=2)
        _write_csv(out / "trajectory.csv", self.trajectory)
        _write_csv(out / "tasks.csv", [{**r, "polygon": json.dumps(r["polygon"]),
                                        "waypoints": json.dumps(r["waypoints"])} for r in self.tasks])
        self.events.to_jsonl(out / "events.jsonl")
        with open(out / "world.json", "w", encoding="utf-8") as f:
            json.dump(self.world, f, ensure_ascii=False)
        with open(out / "scenario.json", "w", encoding="utf-8") as f:
            f.write(self.scenario.canonical_json())
        files = ["metrics.csv", "metrics.json", "trajectory.csv", "tasks.csv", "events.jsonl", "world.json",
                 "scenario.json"]
        if parquet:
            try:
                import pandas as pd

                pd.DataFrame(self.trajectory).to_parquet(out / "trajectory.parquet")
                files.append("trajectory.parquet")
            except Exception as e:  # pragma: no cover - зависит от pyarrow
                self.manifest["parquet_error"] = str(e)
        self.manifest["files"] = files
        with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.manifest), f, ensure_ascii=False, indent=2)
        return out


# =========================================================================== симуляция
class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        allocator: Any = None,
        motion_factory: Optional[MotionFactory] = None,
        sensor: Any = None,
        comms: Any = None,
        faults: Any = None,
        energy: Any = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        sc = scenario
        self.sc, self.seed, self.dt = sc, int(seed), sc.dt
        self._clock = clock or time.perf_counter  # только для метрик wall_*
        streams = np.random.SeedSequence(self.seed).spawn(len(RNG_STREAMS))
        self.rng = {name: np.random.default_rng(s) for name, s in zip(RNG_STREAMS, streams)}

        self.log = EventLog()
        self.world = World.from_config(sc.world, self.rng["world"])
        self.mission = MISSIONS[sc.mission.type]()
        self.tasks: TaskSet = self.mission.decompose(sc.mission, self.world)

        self.sensor = sensor or registry.make_sensor(
            sc.sensor.model, swath_width=sc.sensor.swath_width, pd_pass=sc.sensor.pd_pass, **sc.sensor.params)
        self.energy = energy or registry.make_energy(sc.energy.model, **sc.energy.params)
        if comms is None:
            ctor = registry._resolve(registry.COMMS, sc.comms.model, "comms")
            kw = dict(range=sc.comms.range, delay=sc.comms.delay, jitter=sc.comms.jitter, loss=sc.comms.loss,
                      **sc.comms.params)
            if _accepts(ctor, "rng"):
                kw["rng"] = self.rng["comms"]
            comms = ctor(**kw)
        self.comms = comms
        self.faults = faults or registry.make_faults(sc.faults.model, detect_delay=sc.faults.detect_delay)
        if allocator is None:
            ctor = registry._resolve(registry.ALLOCATORS, sc.allocator.name, "allocator")
            kw = dict(sc.allocator.params)
            if _accepts(ctor, "rng"):
                kw["rng"] = self.rng["alloc"]
            if _accepts(ctor, "scenario"):
                kw["scenario"] = sc
            allocator = ctor(**kw)
        self.allocator = allocator
        self.alloc_name = getattr(allocator, "name", type(allocator).__name__)

        mf = motion_factory or default_motion
        self.agents: dict[int, Agent] = {}
        for i in range(sc.agents.count):
            home = sc.agents.home_of(i)
            alt = sc.agents.altitude + i * sc.agents.altitude_step
            m = mf(i, home, alt, sc.agents)
            m.arm()
            m.takeoff(alt)
            self.agents[i] = Agent(id=i, motion=m, home=home)

        self.metrics = BasicMetrics(sc.agents.min_separation)
        self.schedule = self._build_schedule()
        self._sched_pos = 0
        self.pending: list[Event] = []
        self.failure_event: dict[int, Event] = {}
        self.outages: list[tuple[float, float, str]] = []
        self.trajectory: list[dict[str, Any]] = []
        self._done_logged: set[str] = set()
        self._unassigned_after_alloc: frozenset[str] = frozenset()
        self._last_hb = -math.inf
        self.gcs_last_seen: dict[int, float] = {}
        self.k = 0
        self.t = 0.0
        self._log_every_k = max(1, round(sc.output.log_every / sc.dt))

    # ------------------------------------------------------------------ расписание
    def _build_schedule(self) -> list[tuple[float, int, str, dict[str, Any]]]:
        """Список ``(t, порядок, тип, данные)``; случайные отказы раскрываются зерном."""
        items: list[tuple[float, int, str, dict[str, Any]]] = []
        for k, ev in enumerate(self.sc.events):
            items.append((ev.t, k, ev.type, ev.model_dump(exclude={"t", "type"})))
        rf = self.sc.faults.random_failures
        n_fail = int(round(rf.fraction * self.sc.agents.count))
        if n_fail:
            rng = self.rng["faults"]
            ids = sorted(int(i) for i in rng.choice(self.sc.agents.count, size=n_fail, replace=False))
            t0, t1 = rf.t_window if rf.t_window[1] > rf.t_window[0] else (0.0, self.sc.t_end)
            times = sorted(float(x) for x in rng.uniform(t0, t1, size=n_fail))
            for j, (aid, tf) in enumerate(zip(ids, times)):
                items.append((round(tf / self.dt) * self.dt, 10_000 + j, "agent_failure",
                              {"agent": aid, "random": True}))
        items.sort(key=lambda x: (x[0], x[1]))
        return items

    # ------------------------------------------------------------------ служебное
    def _t_root(self, ev: Optional[Event]) -> Optional[float]:
        if ev is None:
            return None
        root = self.log.root_cause(ev.seq)
        return None if root.type == "mission_start" else root.t

    def gcs_ok(self) -> bool:
        if any(t0 <= self.t < t1 for t0, t1, _ in self.outages):
            return False
        return bool(self.comms.gcs_reachable(self.t))

    def snapshot(self) -> SwarmSnapshot:
        views = {
            i: AgentView(agent_id=i, pos=a.state.pos, home=a.home, alive=a.alive, speed=self.sc.agents.speed,
                         current_task=a.current, queue=list(a.queue), energy_used_wh=a.energy_used_wh)
            for i, a in self.agents.items()
        }
        return SwarmSnapshot(t=self.t, agents=views, world=self.world, gcs_reachable=self.gcs_ok())

    # ------------------------------------------------------------------ распределение
    def allocate(self, cause: Event) -> Assignment:
        w0 = self._clock()
        asg = self.allocator.allocate(self.snapshot(), self.tasks)
        wall = self._clock() - w0
        alloc_ev = self.log.append(type="allocation", t=self.t, source=f"allocator:{self.alloc_name}",
                                   cause=cause.seq, data={"trigger": cause.type})
        owner: dict[str, int] = {}
        for aid in sorted(asg.plans):
            if aid not in self.agents or not self.agents[aid].alive:
                self.log.append(type="allocation_warning", t=self.t, source="sim", cause=alloc_ev.seq,
                                data={"reason": f"план для недоступного агента {aid} отброшен"})
                continue
            for tid in asg.plans[aid]:
                if tid not in self.tasks:
                    continue
                if tid in owner:
                    self.log.append(type="allocation_warning", t=self.t, source="sim", cause=alloc_ev.seq,
                                    data={"task": tid, "reason": f"задача назначена дважды ({owner[tid]}, {aid}); "
                                                                 f"оставлен {owner[tid]}"})
                    continue
                owner[tid] = aid
        plans = {aid: [tid for tid in asg.plans.get(aid, []) if owner.get(tid) == aid] for aid in self.agents}
        prev = {t.id: t.assignee for t in self.tasks.unfinished()}
        for aid in sorted(self.agents):
            if self.agents[aid].alive:
                self.agents[aid].set_plan(plans[aid], self.tasks)
        for tid in sorted(prev):
            old, new = prev[tid], owner.get(tid)
            if old == new:
                continue
            task = self.tasks[tid]
            task.mark_assigned(new, self.t)
            etype = "assign" if old is None else ("unassign" if new is None else "reassign")
            reason = asg.reasons.get(tid) or ("первичное назначение" if cause.type == "mission_start"
                                              else f"по событию {cause.type}")
            self.log.append(type=etype, t=self.t, source=f"allocator:{self.alloc_name}", cause=alloc_ev.seq,
                            data={"task": tid, "from": old, "to": new, "reason": reason})
        self.metrics.on_allocation(self.t, self._t_root(cause), wall)
        self._unassigned_after_alloc = frozenset(t.id for t in self.tasks.unassigned())
        return asg

    # ------------------------------------------------------------------ события
    def _apply_scenario_events(self) -> None:
        while self._sched_pos < len(self.schedule) and self.schedule[self._sched_pos][0] <= self.t + 1e-9:
            _, _, etype, data = self.schedule[self._sched_pos]
            self._sched_pos += 1
            if etype == "agent_failure":
                aid = int(data["agent"])
                if not self.agents[aid].alive:
                    continue
                st = self.agents[aid].state
                ev = self.log.append(type="agent_failure", t=self.t, source="scenario",
                                     data={**data, "pos": [round(st.x, 3), round(st.y, 3)]})
                self.agents[aid].fail()
                self.faults.inject(ev, self.t)
                self.failure_event[aid] = ev
            elif etype == "new_target":
                pos = tuple(data["pos"])
                tg = self.world.add_target(pos[0], pos[1], t_appear=self.t, priority=data.get("priority", 1.0))
                ev = self.log.append(type="new_target", t=self.t, source="scenario",
                                     data={"target": tg.id, "pos": list(pos)})
                if data.get("search_radius", 0) > 0:
                    for task in self.mission.on_new_target(self.sc.mission, self.world, self.tasks, pos,
                                                           data["search_radius"], self.t):
                        tev = self.log.append(type="task_created", t=self.t, source="mission", cause=ev.seq,
                                              data={"task": task.id, "reason": "доразведка вокруг новой цели"})
                        self.pending.append(tev)
            elif etype == "comms_loss":
                t1 = self.t + float(data["duration"])
                scope = data.get("scope", "gcs")
                ev = self.log.append(type="comms_loss", t=self.t, source="scenario",
                                     data={"scope": scope, "until": t1})
                self.outages.append((self.t, t1, scope))
                if hasattr(self.comms, "set_outage"):
                    self.comms.set_outage(self.t, t1, scope)
                self.schedule.append((t1, 20_000 + ev.seq, "comms_restored", {"scope": scope, "loss_seq": ev.seq}))
                self.schedule.sort(key=lambda x: (x[0], x[1]))
            elif etype == "comms_restored":
                ev = self.log.append(type="comms_restored", t=self.t, source="scenario",
                                     data={"scope": data["scope"], "loss_seq": data["loss_seq"]})
                if self.tasks.unassigned():
                    self.pending.append(ev)

    def _detect_failures(self) -> None:
        for aid in self.faults.detect(self.t, self.agents):
            cause = self.failure_event.get(aid)
            ev = self.log.append(type="failure_detected", t=self.t, source="faults",
                                 cause=cause.seq if cause else None, data={"agent": aid})
            released = self.agents[aid].release_all(self.tasks)
            released = [tid for tid in released if not self.tasks[tid].is_finished]
            # задачи, которые числились за агентом, но не попали в его очередь
            for t in self.tasks.unfinished():
                if t.assignee == aid and t.id not in released:
                    released.append(t.id)
            for tid in released:
                self.tasks[tid].mark_assigned(None, self.t)
                self.log.append(type="task_released", t=self.t, source="sim", cause=ev.seq,
                                data={"task": tid, "from": aid, "to": None, "reason": f"отказ агента {aid}"})
            t_fail = cause.t if cause else self.t
            self.metrics.on_failure_detected(aid, t_fail, self.t, released)
            self.pending.append(ev)

    def _comms_step(self) -> list[Message]:
        if self.t - self._last_hb >= self.sc.comms.heartbeat_period - 1e-9:
            self._last_hb = self.t
            for aid in sorted(self.agents):
                a = self.agents[aid]
                if a.alive:
                    self.comms.send(aid, GCS_ID, Message(src=aid, dst=GCS_ID, kind="heartbeat", t_sent=self.t,
                                                         payload={"pos": a.state.pos, "task": a.current}))
        pos = {i: a.state.pos for i, a in self.agents.items()}
        pos[GCS_ID] = self.sc.agents.base
        msgs = sorted(self.comms.step(self.t, pos), key=lambda m: (m.t_deliver, m.src, m.seq))
        on_msg = getattr(self.allocator, "on_message", None)
        for m in msgs:
            if m.dst == GCS_ID and m.kind == "heartbeat":
                self.gcs_last_seen[m.src] = m.t_sent
            if on_msg is not None:
                on_msg(m)
        return msgs

    # ------------------------------------------------------------------ шаг
    def start(self) -> None:
        ev = self.log.append(type="mission_start", t=0.0, source="sim",
                             data={"n_agents": len(self.agents), "n_tasks": len(self.tasks),
                                   "allocator": self.alloc_name, "seed": self.seed})
        self.allocate(ev)

    def step(self) -> None:
        w0 = self._clock()
        self._apply_scenario_events()
        self._detect_failures()
        self._comms_step()

        gcs = self.gcs_ok()
        unassigned = frozenset(t.id for t in self.tasks.unassigned())
        if self.pending or (unassigned and unassigned != self._unassigned_after_alloc):
            trig_fn = getattr(self.allocator, "should_trigger", None)
            trig = trig_fn(self.t, self.tasks, list(self.pending)) if trig_fn else bool(self.pending)
            needs_gcs = getattr(self.allocator, "requires_gcs", True)
            if trig and (gcs or not needs_gcs):
                cause = self.pending[0] if self.pending else self.log.append(
                    type="sim_trigger", t=self.t, source="sim", data={"reason": "есть неназначенные задачи"})
                self.allocate(cause)
                self.pending = []

        for aid in sorted(self.agents):
            a = self.agents[aid]
            st = a.step(self.dt, self.tasks, self.world)
            if a.alive and st.alt > 0:
                a.energy_used_wh += self.energy.power(st.speed) * self.dt / 3600.0
            det = self.world.observe(a, self.sensor)
            for tgid in det.new_detections:
                self.log.append(type="target_detected", t=self.t, source=f"agent:{aid}", data={"target": tgid})
        for task in self.tasks:
            if task.status == TaskStatus.DONE and task.id not in self._done_logged:
                self._done_logged.add(task.id)
                self.log.append(type="task_done", t=self.t + self.dt, source=f"agent:{task.done_by}",
                                data={"task": task.id})
                self.metrics.on_task_done(gcs)

        self.world.step(self.dt)
        self.k += 1
        self.t = self.k * self.dt
        self.metrics.record(self.t, self.world, self.agents, self.tasks, self.comms, gcs_ok=gcs, dt=self.dt)
        if self.k % self._log_every_k == 0:
            self._log_trajectory()
        self.metrics.wall_step_s += self._clock() - w0

    def _log_trajectory(self) -> None:
        for aid in sorted(self.agents):
            a = self.agents[aid]
            s = a.state
            self.trajectory.append({"t": round(self.t, 6), "agent": aid, "x": round(s.x, 3), "y": round(s.y, 3),
                                    "alt": round(s.alt, 2), "alive": int(a.alive), "status": a.status,
                                    "task": a.current or "", "gcs": int(self.gcs_ok())})

    def finished(self) -> bool:
        if not any(a.alive for a in self.agents.values()):
            return True
        if not self.sc.stop_when_done:
            return False
        pending_sched = any(s[2] != "comms_restored" for s in self.schedule[self._sched_pos:])
        return (not self.tasks.unfinished() and not pending_sched
                and all(a.status in ("home", "idle") for a in self.agents.values() if a.alive))

    @property
    def n_steps(self) -> int:
        return int(round(self.sc.t_end / self.dt))

    def run(self, pace: Optional[Callable[["Simulation"], None]] = None) -> RunResult:
        """Полный прогон. ``pace`` вызывается после каждого шага (например, для
        синхронизации с реальным временем в SITL) и не влияет на логику цикла."""
        self.start()
        self._log_trajectory()
        while self.k < self.n_steps and not self.finished():
            self.step()
            if pace is not None:
                pace(self)
        return self.finish()

    def finish(self) -> RunResult:
        self.log.append(type="mission_end", t=self.t, source="sim",
                        data={"done": len(self.tasks.by_status(TaskStatus.DONE)), "total": len(self.tasks)})
        metrics = self.metrics.finalize(self.world, self.agents, self.tasks, self.comms, self.sc.t_end)
        for m in self.agents.values():
            m.motion.close()
        res = RunResult(scenario=self.sc, seed=self.seed, allocator=self.alloc_name, metrics=metrics,
                        trajectory=self.trajectory, tasks=[t.to_row() for t in self.tasks], events=self.log,
                        world=self.world.snapshot())
        res.manifest = build_manifest(res)
        return res


def run(scenario: Scenario | str | Path, seed: int, **kw: Any) -> RunResult:
    """``run(scenario, seed) -> RunResult`` — точка входа стенда."""
    if not isinstance(scenario, Scenario):
        scenario = Scenario.from_yaml(scenario)
    return Simulation(scenario, seed, **kw).run()


# =========================================================================== manifest
def build_manifest(res: RunResult) -> dict[str, Any]:
    return {
        "format": "swarmcore-run",
        "version": 1,
        "scenario": res.scenario.name,
        "config_hash": res.scenario.config_hash(),
        "seed": res.seed,
        "allocator": res.allocator,
        "metrics_hash": res.metrics_hash(),
        "events_hash": res.events_hash(),
        "swarmcore_version": swarmsim.__version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _hash_obj(obj: Any) -> str:
    s = json.dumps(_jsonable(obj), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()})
