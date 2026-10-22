"""Визуализация прогона (M2-10) — строго из сохранённых журналов прогона.

Источник данных — только файлы папки прогона: ``trajectory.csv``, ``events.jsonl``,
``tasks.csv``, ``world.json``. Никакой собственной логики симуляции здесь нет:
визуализация — производная от журнала, а не наоборот.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from swarmsim.eventlog import EventLog

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
           "#7f7f7f", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d"]


def _color(aid: Any) -> str:
    return PALETTE[int(aid) % len(PALETTE)] if aid not in (None, "", "None") else "#dddddd"


class RunData:
    def __init__(self, run_dir: str | Path) -> None:
        d = Path(run_dir)
        self.dir = d
        with open(d / "world.json", encoding="utf-8") as f:
            self.world = json.load(f)
        self.log = EventLog.from_jsonl(d / "events.jsonl")
        self.traj: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with open(d / "trajectory.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.traj[int(r["agent"])].append(
                    {"t": float(r["t"]), "x": float(r["x"]), "y": float(r["y"]), "alive": r["alive"] == "1",
                     "task": r["task"], "status": r["status"]})
        self.times = sorted({p["t"] for rows in self.traj.values() for p in rows})
        self.tasks: dict[str, dict[str, Any]] = {}
        with open(d / "tasks.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["polygon"] = json.loads(r["polygon"])
                self.tasks[r["id"]] = r
        # история владельцев задач из журнала
        self.owner_hist: dict[str, list[tuple[float, Any]]] = defaultdict(list)
        self.done_t: dict[str, float] = {}
        self.created_t: dict[str, float] = {}
        for e in self.log:
            tid = e.data.get("task")
            if e.type in ("assign", "reassign", "unassign", "task_released"):
                self.owner_hist[tid].append((e.t, e.data.get("to")))
            elif e.type == "task_done":
                self.done_t[tid] = e.t
            elif e.type == "task_created":
                self.created_t[tid] = e.t
        self.failures = [(e.t, e.data["agent"], e.data.get("pos")) for e in self.log.of_type("agent_failure")]
        self.outages = [(e.t, e.data["until"], e.data["scope"]) for e in self.log.of_type("comms_loss")]
        self.notes = [e for e in self.log.of_type("failure_detected", "reassign", "task_released", "comms_loss",
                                                 "comms_restored", "new_target", "task_created")]
        # повторные назначения после первичного — для подписи «перераспределение»
        self.realloc = [e for e in self.log.of_type("assign", "reassign")
                        if self.log.root_cause(e.seq).type != "mission_start"]

    def owner_at(self, tid: str, t: float) -> Any:
        own = None
        for tt, o in self.owner_hist.get(tid, []):
            if tt <= t:
                own = o
        return own

    def pos_at(self, aid: int, t: float) -> dict[str, Any] | None:
        rows = self.traj[aid]
        lo, hi = 0, len(rows) - 1
        if not rows or rows[0]["t"] > t:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if rows[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid - 1
        return rows[lo]


def _draw_static(ax, rd: RunData) -> None:
    from matplotlib.patches import Polygon as MplPolygon

    b = rd.world["bounds"]
    ax.add_patch(MplPolygon(b, closed=True, fill=False, ec="black", lw=1.2))
    for z in rd.world["nofly"]:
        ax.add_patch(MplPolygon(z, closed=True, fc="#f4c7c3", ec="#c0392b", hatch="//", lw=1, zorder=1))
    xs, ys = zip(*b)
    pad = 0.04 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м (восток)")
    ax.set_ylabel("y, м (север)")


def animate(run_dir: str | Path, out: str | Path, fps: int = 15, speed: float = 20.0, trail_s: float = 120.0,
            max_frames: int = 400) -> str:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon as MplPolygon

    rd = RunData(run_dir)
    t_end = rd.times[-1]
    step = max(speed / fps, t_end / max_frames)
    frames = []
    t = 0.0
    while t <= t_end + 1e-9:
        frames.append(t)
        t += step
    if frames[-1] < t_end:
        frames.append(t_end)

    fig = plt.figure(figsize=(10, 8.4))
    ax = fig.add_axes([0.07, 0.24, 0.62, 0.72])
    ax_side = fig.add_axes([0.71, 0.24, 0.28, 0.72])
    ax_tl = fig.add_axes([0.07, 0.06, 0.92, 0.1])
    ax_side.axis("off")
    _draw_static(ax, rd)

    cell_patches = {}
    for tid, task in rd.tasks.items():
        if len(task["polygon"]) >= 3:
            p = MplPolygon(task["polygon"], closed=True, fc="#eeeeee", ec="#bbbbbb", lw=0.5, alpha=0.45, zorder=0)
            ax.add_patch(p)
            cell_patches[tid] = p
    tg = rd.world["targets"]
    tg_sc = ax.scatter([t["x"] for t in tg], [t["y"] for t in tg], marker="^", s=28, c="white",
                       edgecolors="black", linewidths=0.6, zorder=3)
    agents = sorted(rd.traj)
    arr = {a: (np.array([r["t"] for r in rd.traj[a]]), np.array([r["x"] for r in rd.traj[a]]),
               np.array([r["y"] for r in rd.traj[a]]), np.array([r["alive"] for r in rd.traj[a]]))
           for a in agents}
    trails = {a: ax.plot([], [], "-", color=_color(a), lw=1.4, alpha=0.9, zorder=4)[0] for a in agents}
    paths = {a: ax.plot([], [], "-", color=_color(a), lw=0.5, alpha=0.35, zorder=2)[0] for a in agents}
    dots = {a: ax.plot([], [], "o", color=_color(a), ms=7, mec="black", mew=0.6, zorder=6)[0] for a in agents}
    labels = {a: ax.text(0, 0, f"A{a}", fontsize=7, zorder=7) for a in agents}
    fail_marks = ax.scatter([], [], marker="X", s=140, c="red", edgecolors="black", zorder=8)
    banner = ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=11,
                     color="#c0392b", fontweight="bold")
    title = ax.set_title("")

    # шкала времени: интервалы потери связи и моменты отказов
    ax_tl.set_xlim(0, t_end)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_yticks([])
    ax_tl.set_xlabel("модельное время, с")
    for t0, t1, scope in rd.outages:
        ax_tl.axvspan(t0, min(t1, t_end), color="#e74c3c" if scope == "all" else "#f39c12", alpha=0.45,
                      label=f"потеря связи ({'вся' if scope == 'all' else 'с НС'})")
    for tf, aid, _ in rd.failures:
        ax_tl.axvline(tf, color="red", lw=2)
        ax_tl.text(tf, 0.55, f" отказ A{aid}", color="red", fontsize=8)
    for e in rd.realloc:
        ax_tl.plot([e.t], [0.2], "|", color="black", ms=10)
    if rd.outages:
        h, lab = ax_tl.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        ax_tl.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=7)
    cursor = ax_tl.axvline(0, color="black", lw=1)
    side = ax_side.text(0, 1, "", va="top", fontsize=8, family="monospace")

    def update(t: float):
        for tid, p in cell_patches.items():
            if tid in rd.created_t and rd.created_t[tid] > t:
                p.set_visible(False)
                continue
            p.set_visible(True)
            if tid in rd.done_t and rd.done_t[tid] <= t:
                p.set_facecolor(_color(rd.owner_at(tid, t)))
                p.set_alpha(0.25)
            else:
                own = rd.owner_at(tid, t)
                p.set_facecolor(_color(own) if own is not None else "#ff4d4d")
                p.set_alpha(0.12 if own is not None else 0.35)
        cols = []
        for x in tg:
            if x["detected_t"] is not None and x["detected_t"] <= t:
                cols.append("#27ae60")
            elif x["t_appear"] <= t:
                cols.append("white")
            else:
                cols.append((0, 0, 0, 0))
        tg_sc.set_facecolors(cols)
        for a in agents:
            T, X, Y, AL = arr[a]
            n = int(np.searchsorted(T, t, side="right"))
            if n == 0:
                continue
            m = int(np.searchsorted(T, t - trail_s, side="left"))
            trails[a].set_data(X[m:n], Y[m:n])
            paths[a].set_data(X[:n], Y[:n])
            dots[a].set_data([X[n - 1]], [Y[n - 1]])
            dots[a].set_alpha(1.0 if AL[n - 1] else 0.3)
            labels[a].set_position((X[n - 1] + 8, Y[n - 1] + 8))
        fx = [(p[0], p[1]) for tf, _, p in rd.failures if tf <= t and p]
        fail_marks.set_offsets(fx if fx else [[float("nan"), float("nan")]])
        lost = [o for o in rd.outages if o[0] <= t < o[1]]
        banner.set_text(("НЕТ СВЯЗИ " + ("(вся сеть)" if lost[0][2] == "all" else "С НАЗЕМНОЙ СТАНЦИЕЙ"))
                        if lost else "")
        done = sum(1 for v in rd.done_t.values() if v <= t)
        title.set_text(f"{rd.dir.parent.name}/{rd.dir.name}   t = {t:6.0f} с   задач выполнено {done}/{len(rd.tasks)}")
        cursor.set_xdata([t, t])
        lines = ["Журнал (последние события):", ""]
        recent = [e for e in rd.notes if e.t <= t][-14:]
        for e in recent:
            d = e.data
            if e.type in ("reassign", "task_released"):
                s = f"{e.t:6.0f} {'перед.' if e.type == 'reassign' else 'освоб.'} {d.get('task', '').replace('cell_', '')} {d.get('from')}→{d.get('to')}"
            elif e.type == "failure_detected":
                s = f"{e.t:6.0f} ОТКАЗ обнаружен A{d.get('agent')}"
            elif e.type == "comms_loss":
                s = f"{e.t:6.0f} потеря связи ({d.get('scope')})"
            elif e.type == "comms_restored":
                s = f"{e.t:6.0f} связь восстановлена"
            elif e.type == "new_target":
                s = f"{e.t:6.0f} новая цель #{d.get('target')}"
            else:
                s = f"{e.t:6.0f} {e.type} {d.get('task', '')}"
            lines.append(s)
        realloc_now = [e for e in rd.realloc if e.t <= t]
        lines += ["", f"перераспределений: {len(realloc_now)}"]
        for e in realloc_now[-6:]:
            lines.append(f"{e.t:6.0f} {e.data['task'].replace('cell_', '')} → A{e.data['to']}")
        side.set_text("\n".join(lines))
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = Path(out)
    if out.suffix.lower() == ".mp4":
        anim.save(out, writer=FFMpegWriter(fps=fps))
    else:
        anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(out)


def plot_summary(run_dir: str | Path, out: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    rd = RunData(run_dir)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    _draw_static(ax, rd)
    for tid, task in rd.tasks.items():
        if len(task["polygon"]) >= 3:
            own = task.get("done_by") or rd.owner_at(tid, rd.times[-1])
            ax.add_patch(MplPolygon(task["polygon"], closed=True, fc=_color(own), alpha=0.15, ec="#999999", lw=0.4))
    for a, rows in sorted(rd.traj.items()):
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], "-", color=_color(a), lw=0.8, label=f"A{a}")
    for tf, aid, p in rd.failures:
        if p:
            ax.plot([p[0]], [p[1]], "X", color="red", ms=13, mec="black")
            ax.annotate(f"отказ A{aid}\nt={tf:.0f} с", p, xytext=(8, 8), textcoords="offset points", color="red")
    tg = rd.world["targets"]
    ax.scatter([t["x"] for t in tg], [t["y"] for t in tg], marker="^", s=30,
               c=["#27ae60" if t["detected_t"] is not None else "white" for t in tg], edgecolors="black", zorder=5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_title("Траектории, отказы, цели (зелёные — обнаружены)")

    t_end = rd.times[-1]
    done = sorted(rd.done_t.values())
    ax2.step([0] + done + [t_end], [0] + list(range(1, len(done) + 1)) + [len(done)], where="post",
             label="выполнено задач")
    det = sorted(t["detected_t"] for t in tg if t["detected_t"] is not None)
    ax2.step([0] + det + [t_end], [0] + list(range(1, len(det) + 1)) + [len(det)], where="post",
             label="обнаружено целей")
    for t0, t1, scope in rd.outages:
        ax2.axvspan(t0, min(t1, t_end), color="#f39c12", alpha=0.3, label="потеря связи")
    for tf, aid, _ in rd.failures:
        ax2.axvline(tf, color="red", ls="--", label=f"отказ A{aid}")
    for e in rd.realloc:
        ax2.axvline(e.t, color="black", lw=0.5, alpha=0.5)
    h, lab = ax2.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax2.legend(uniq.values(), uniq.keys(), fontsize=8)
    ax2.set_xlabel("модельное время, с")
    ax2.set_title("Ход миссии (тонкие линии — перераспределения)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out)
