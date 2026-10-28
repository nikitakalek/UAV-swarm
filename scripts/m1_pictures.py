"""Картинки и цифры модуля 1 для документов и заявки.

    ./.venv/bin/python scripts/m1_pictures.py

Результат в docs/img/:
* ``sectors_comparison.png`` — Вороной + Ллойд против power diagram (разброс площадей);
* ``plan_validation.png`` — план группы и нарушения валидатора: все на одной высоте
  против эшелонирования;
и таблица времени расчёта ``SectorAllocator`` в консоли.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from shapely.geometry import box  # noqa: E402

from swarmsim.alloc import SectorAllocator  # noqa: E402
from swarmsim.planning.partition import area_stats, build_partition  # noqa: E402
from swarmsim.planning.validator import PlanValidator  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from m1_helpers import make_world_tasks, snapshot  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
COLORS = plt.get_cmap("tab10").colors


def _draw_geom(ax, geom, color, alpha=0.35):
    for g in getattr(geom, "geoms", [geom]):
        if not g.is_empty:
            ax.add_patch(MplPolygon(list(g.exterior.coords), closed=True, facecolor=(*color[:3], alpha),
                                    edgecolor="black", lw=0.8))


def sectors_comparison() -> Path:
    domain = box(0, 0, 1000, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (title, power) in zip(axes, [("Вороной + Ллойд", 0), ("+ power diagram (используется)", 60)]):
        part = build_partition(domain, 7, power_iters=power)
        st = area_stats(part)
        for k, s in enumerate(part.sectors):
            _draw_geom(ax, s, COLORS[k % 10])
        ax.set_title(f"{title}\nmax/min площади = {st['ratio']:.3f}")
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect("equal")
    fig.suptitle("Деление участка на 7 компактных секторов")
    fig.tight_layout()
    path = OUT / "sectors_comparison.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plan_validation() -> Path:
    zone = [(420, 420), (580, 420), (580, 580), (420, 580)]
    world, tasks = make_world_tasks(n_cells=36, nofly=[zone])
    pos = {i: (i * 5.0, 0.0) for i in range(5)}
    snap = snapshot(world, pos)
    asg = SectorAllocator().allocate(snap, tasks)
    v = PlanValidator(battery_wh=400)
    same = v.validate(asg, snap, tasks, {i: 40.0 for i in pos})
    layered = v.validate(asg, snap, tasks, {i: 40.0 + 4 * i for i in pos})

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (title, rep) in zip(axes, [("все на 40 м", same), ("эшелоны 40/44/48/52/56 м", layered)]):
        for aid, plan in asg.plans.items():
            for tid in plan:
                ax.add_patch(MplPolygon(tasks[tid].polygon, closed=True,
                                        facecolor=(*COLORS[aid % 10][:3], 0.15), edgecolor="white"))
            xs, ys = zip(*rep.paths[aid])
            ax.plot(xs, ys, color=COLORS[aid % 10], lw=0.9)
        ax.add_patch(MplPolygon(zone, closed=True, facecolor="#999", alpha=0.7))
        for viol in rep.violations:
            if viol.location:
                ax.plot(*viol.location, "x", color="red", ms=9, mew=2)
        ax.set_title(f"{title}: нарушений {len(rep.violations)} {rep.by_rule() or ''}")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 1020)
        ax.set_ylim(-20, 1020)
    fig.suptitle("План 5 аппаратов с одной площадки и проверка валидатором (красное — нарушения)")
    fig.tight_layout()
    path = OUT / "plan_validation.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  одна высота: {same.by_rule()}; эшелоны: {layered.by_rule() or 'нарушений нет'}")
    return path


def timing() -> None:
    print("Время расчёта SectorAllocator (начальный план), задач / аппаратов → мс")
    for cells, n in [(36, 3), (64, 5), (100, 10), (225, 20)]:
        world, tasks = make_world_tasks(n_cells=cells)
        snap = snapshot(world, {i: (i * 5.0, 0.0) for i in range(n)})
        t0 = time.perf_counter()
        SectorAllocator().allocate(snap, tasks)
        print(f"  {cells:4d} / {n:2d}: {1000 * (time.perf_counter() - t0):7.0f}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(sectors_comparison())
    print(plan_validation())
    timing()
