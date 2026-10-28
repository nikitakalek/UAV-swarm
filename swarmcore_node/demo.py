"""Демо MVP на одной машине: N процессов-узлов по UDP + наземная станция, отказ, отчёт.

Сценарий (docs/05): оператор одной командой даёт участок группе → узлы делят его на
секторы и облетают «змейкой» на разных высотах → один аппарат отказывает (процесс узла
убивается) → оставшиеся замечают потерю и доделывают его часть → все садятся.

Результат в ``--out``: журналы узлов и станции, ``summary.json`` с проверками критериев
готовности MVP и картинка ``demo.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from swarmsim.scenario import load_scenario


def run_demo(args) -> int:
    from swarmcore_node.__main__ import cmd_gcs

    sc = load_scenario(args.mission)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("node_*"):
        f.unlink()
    n = sc.agents.count
    epoch = time.time()
    common = ["--n", str(n), "--time-scale", str(args.time_scale), "--epoch", repr(epoch),
              "--port-base", str(args.port_base), "--gcs", f"127.0.0.1:{args.gcs_port}", "--out", str(out)]
    procs = {}
    for i in range(n):
        hx, hy = sc.agents.home_of(i)
        cmd = [sys.executable, "-m", "swarmcore_node", "node", "--id", str(i), "--home", f"{hx},{hy}",
               "--speed", str(sc.agents.speed), "--vehicle", args.vehicle,
               "--altitude", str(sc.agents.altitude + i * sc.agents.altitude_step), *common]
        if getattr(args, "vehicle_commands", None):
            cmd += ["--vehicle-commands", args.vehicle_commands]
        procs[i] = subprocess.Popen(cmd, stdout=open(out / f"node_{i}.stdout", "w"), stderr=subprocess.STDOUT)

    def on_fail(i: int) -> None:
        if args.fail_mode == "kill" and i in procs:
            procs[i].send_signal(signal.SIGKILL)  # отказ борта: узел исчезает без «прощания»

    fails = [] if args.fail in (None, "none") else [args.fail]
    gargs = argparse.Namespace(
        mission=args.mission, n=None, host="127.0.0.1", port_base=args.port_base, peers=None,
        gcs=f"127.0.0.1:{args.gcs_port}", time_scale=args.time_scale, epoch=epoch, loop_dt=0.05,
        out=str(out), fail=fails, rtl_at=None, silence=None, timeout=args.timeout, print_every=20.0,
        quiet=args.quiet, on_fail=on_fail)
    try:
        cmd_gcs(gargs)
    finally:
        time.sleep(0.5)
        for p in procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    summary = analyze(out)
    print(json.dumps(summary["checks"], ensure_ascii=False, indent=1))
    try:
        plot(out, summary)
        print(f"картинка: {out / 'demo.png'}")
    except ImportError:
        print("matplotlib не установлен — картинка пропущена")
    return 0 if all(summary["checks"].values()) else 1


# ----------------------------------------------------------------------------- анализ
def _load(out: Path) -> tuple[dict, list[dict], dict[int, list[dict]]]:
    meta = json.loads((out / "gcs_events.json").read_text())
    hbs = [json.loads(line) for line in (out / "gcs_heartbeats.jsonl").read_text().splitlines() if line]
    events: dict[int, list[dict]] = {}
    for f in sorted(out.glob("node_*_events.jsonl")):
        i = int(f.name.split("_")[1])
        rows = [json.loads(line) for line in f.read_text().splitlines() if line]
        events[i] = [r for r in rows if "type" in r]  # первая строка — заголовок формата
    return meta, hbs, events


def analyze(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    meta, hbs, events = _load(out)
    n_tasks = max((h.get("n_tasks", 0) for h in hbs), default=0)
    done = set()
    for h in hbs:
        done.update(h.get("done", []))
    done_by = Counter(e["data"]["task"] for ev in events.values() for e in ev if e["type"] == "task_done")
    fails = {int(e["cmd"].split(":")[1]): e["t"] for e in meta["events"] if e.get("cmd", "").startswith("fail:")}
    plans = sorted((e for ev in events.values() for e in ev if e["type"] == "plan_published"), key=lambda e: e["t"])
    realloc = {}
    for i, tf in fails.items():
        after = [e["t"] for e in plans if e["t"] >= tf and i in e["data"].get("gone", [])]
        realloc[i] = round(min(after) - tf, 2) if after else None

    # минимальный разнос по высоте между летящими аппаратами (по ближайшим по времени heartbeat)
    alt_now: dict[int, tuple[float, float, bool]] = {}
    min_dz = math.inf
    for h in hbs:
        alt_now[h["src"]] = (h["t"], h.get("alt", 0.0), bool(h.get("flying")))
        flying = [(a, t) for s, (t, a, f) in alt_now.items() if f and h["t"] - t < 1.5]
        for k in range(len(flying)):
            for m in range(k + 1, len(flying)):
                min_dz = min(min_dz, abs(flying[k][0] - flying[m][0]))

    checks = {
        "все задачи выполнены": n_tasks > 0 and len(done) == n_tasks,
        "нет дублей": all(c == 1 for c in done_by.values()),
        "перераспределение ≤ 5 с": all(v is not None and v <= 5.0 for v in realloc.values()),
        "разнос по высоте ≥ 3 м": min_dz >= 3.0,
    }
    summary = {
        "mission_id": meta["mission_id"], "n_tasks": n_tasks, "done": len(done),
        "duplicates": sorted(t for t, c in done_by.items() if c > 1), "fail_t": fails, "realloc_s": realloc,
        "min_vertical_sep_m": None if math.isinf(min_dz) else round(min_dz, 2),
        "replans": [{"t": e["t"], "leader": int(e["source"].split(":")[1]), **{k: e["data"][k] for k in
                     ("version", "cause", "members")}} for e in plans],
        "t_end": max((h["t"] for h in hbs), default=0.0), "checks": checks,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


# ----------------------------------------------------------------------------- картинка
def plot(out: str | Path, summary: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon as MplPolygon

    from swarmsim.missions import decompose
    from swarmsim.scenario import Scenario
    from swarmsim.world import World

    out = Path(out)
    meta, hbs, events = _load(out)
    sc = Scenario.model_validate(meta["scenario"])
    world = World.from_config(sc.world, np.random.default_rng(0))
    tasks = decompose(sc.mission, world)
    colors = plt.get_cmap("tab10").colors

    owner0: dict[str, int] = {}
    moved: dict[str, int] = {}
    for ev in events.values():
        for e in ev:
            if e["type"] != "assign":
                continue
            if e["data"]["from"] is None and e["data"]["task"] not in owner0:
                owner0[e["data"]["task"]] = e["data"]["to"]
            elif e["data"]["from"] is not None:
                moved[e["data"]["task"]] = e["data"]["to"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    for t in tasks:
        own = owner0.get(t.id)
        c = colors[own % 10] if own is not None else (0.85, 0.85, 0.85)
        ax.add_patch(MplPolygon(t.polygon, closed=True, facecolor=(*c[:3], 0.18), edgecolor="white", lw=1))
        if t.id in moved:
            ax.add_patch(MplPolygon(t.polygon, closed=True, fill=False, hatch="//",
                                    edgecolor=(*colors[moved[t.id] % 10][:3], 0.8), lw=0))
    for z in world.nofly:
        ax.add_patch(MplPolygon(z, closed=True, facecolor="#999", edgecolor="#555", alpha=0.6))
    bx, by = zip(*(list(world.bounds) + [world.bounds[0]]))
    ax.plot(bx, by, color="black", lw=1)

    by_src: dict[int, list[dict]] = {}
    for h in hbs:
        by_src.setdefault(h["src"], []).append(h)
    for i, rows in sorted(by_src.items()):
        xs, ys = [r["pos"][0] for r in rows], [r["pos"][1] for r in rows]
        alt = sc.agents.altitude + i * sc.agents.altitude_step
        ax.plot(xs, ys, color=colors[i % 10], lw=1.4, label=f"аппарат {i} ({alt:.0f} м)")
        if i in {int(k) for k in summary["fail_t"]}:
            ax.plot(xs[-1], ys[-1], "X", color="red", ms=16, mec="black")
            ax.annotate(f"отказ {i}", (xs[-1], ys[-1]), xytext=(8, 8), textcoords="offset points", color="red")
    ax.set_aspect("equal")
    ax.set_title("Секторы (заливка), переданные после отказа (штриховка), траектории")
    ax.set_xlabel("x, м (восток)")
    ax.set_ylabel("y, м (север)")
    ax.legend(loc="upper right", fontsize=8)

    ts, nd = [], []
    seen: set[str] = set()
    for h in hbs:
        seen.update(h.get("done", []))
        ts.append(h["t"])
        nd.append(len(seen))
    ax2.step(ts, nd, where="post", label="выполнено задач", color="tab:blue")
    for i, tf in summary["fail_t"].items():
        ax2.axvline(tf, color="red", ls="--", label=f"отказ аппарата {i}")
        r = summary["realloc_s"].get(i) if isinstance(summary["realloc_s"], dict) else None
        r = r if r is not None else summary["realloc_s"].get(str(i))
        if r is not None:
            ax2.axvline(tf + r, color="green", ls=":", label=f"новый план через {r:.1f} с")
    ax2.axhline(summary["n_tasks"], color="gray", lw=0.8)
    ax2.set_xlabel("время, с")
    ax2.set_ylabel("задач")
    ok = all(summary["checks"].values())
    ax2.set_title("Ход миссии — " + ("все проверки пройдены" if ok else "есть непройденные проверки"))
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "demo.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
