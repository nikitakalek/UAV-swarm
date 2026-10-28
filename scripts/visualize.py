"""Визуализация работы Модуля 1.

Два режима:

* без аргументов — сравнение разбиения области (полосы / CVT / power diagram);
* ``--plan`` — полный пайплайн: секторы → маршруты → проверка плана, два варианта
  высот (все на одной и эшелонирование). Красные крестики — нарушения.

Запуск:
    ./.venv/bin/python scripts/visualize.py
    ./.venv/bin/python scripts/visualize.py --plan

Результат:
    docs/img/sectors_comparison.png
    docs/img/plan_validation.png
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Каталог ~/.matplotlib может быть недоступен для записи — уводим кэш во временный каталог.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

import matplotlib

matplotlib.use("Agg")  # без графического окна: сохраняем в файл
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import box

from swarmsim.alloc import StaticNaiveAllocator
from swarmsim.alloc import StaticVoronoiAllocator
from swarmsim.plan import PlanValidator, route_assignment
from swarmsim.sectors import area_stats

AREA = 1000.0
N_AGENTS = 10
GRID = 25  # ячеек задач по стороне
OUT_DIR = Path(__file__).parent.parent / "docs" / "img"
COLORS = ("#d62728", "#ff7f0e", "#1f77b4")


# --- заглушки данных (в реальной системе их дают Модуль 2) ------------------


@dataclass(frozen=True)
class StubTask:
    id: int
    centroid: tuple[float, float]


@dataclass(frozen=True)
class StubState:
    agent_ids: tuple[int, ...]
    _domain: object

    def domain_polygon(self):
        return self._domain


def make_tasks(nx: int = GRID, ny: int = GRID, size: float = AREA) -> list[StubTask]:
    step = size / nx
    return [
        StubTask(iy * nx + ix, (step * (ix + 0.5), step * (iy + 0.5)))
        for iy in range(ny)
        for ix in range(nx)
    ]


# --- отрисовка -------------------------------------------------------------


def draw_sectors(ax, partition, title: str) -> None:
    cmap = plt.get_cmap("tab20")
    for i, sector in enumerate(partition.sectors):
        if sector.is_empty:
            continue
        parts = sector.geoms if sector.geom_type == "MultiPolygon" else [sector]
        for g in parts:
            xs, ys = g.exterior.xy
            ax.add_patch(
                MplPolygon(
                    list(zip(xs, ys)),
                    closed=True,
                    facecolor=cmap(i % 20),
                    edgecolor="black",
                    alpha=0.35,
                    linewidth=0.8,
                )
            )
    ax.set_title(title)
    ax.set_xlim(0, AREA)
    ax.set_ylim(0, AREA)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.3)


def draw_tasks(ax, tasks, assignment) -> None:
    by_id = {t.id: t for t in tasks}
    cmap = plt.get_cmap("tab20")
    for agent, task_ids in assignment.by_agent.items():
        pts = [by_id[tid] for tid in task_ids]
        if not pts:
            continue
        ax.scatter(
            [p.centroid[0] for p in pts],
            [p.centroid[1] for p in pts],
            s=5,
            color=cmap(agent % 20),
            zorder=3,
        )


def elongation(sector) -> float:
    return sector.length / (4.0 * (sector.area ** 0.5))


def draw_areas(ax, results) -> None:
    """Отклонение площади сектора от идеала в процентах — честнее абсолютных значений."""
    ideal = AREA * AREA / N_AGENTS
    for (name, partition) in results:
        areas = sorted(a for a in partition.areas if a > 0)
        dev = [(a / ideal - 1.0) * 100.0 for a in areas]
        ax.plot(range(1, len(dev) + 1), dev, marker="o", markersize=3, label=name)
    ax.axhline(0.0, ls="--", color="gray", label="идеал")
    ax.set_title("Отклонение площади сектора от идеала, %")
    ax.set_xlabel("сектор (отсортировано)")
    ax.set_ylabel("отклонение, %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def draw_shape(ax, results) -> None:
    names = [name for name, _ in results]
    elong = [max(elongation(s) for s in p.sectors if not s.is_empty) for _, p in results]
    perim = [sum(s.length for s in p.sectors) for _, p in results]

    ax.bar(names, elong, color=COLORS[: len(names)])
    for i, (e, pe) in enumerate(zip(elong, perim)):
        ax.text(
            i,
            e + 0.06,
            f"{e:.2f}\nпериметр {pe:,.0f}".replace(",", " "),
            ha="center",
            fontsize=8,
        )
    ax.axhline(1.0, ls=":", color="gray")
    ax.set_ylim(0, max(elong) * 1.4)
    ax.set_title("Вытянутость секторов (1.0 = квадрат)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")


def main() -> None:
    domain = box(0.0, 0.0, AREA, AREA)
    tasks = make_tasks()
    state = StubState(tuple(range(N_AGENTS)), domain)

    naive = StaticNaiveAllocator()
    naive_assign = naive.allocate(state, tasks)

    cvt = StaticVoronoiAllocator(lloyd_iters=100, power_iters=0)
    cvt.allocate(state, tasks)

    power = StaticVoronoiAllocator(lloyd_iters=50, power_iters=60)
    power_assign = power.allocate(state, tasks)

    results = [
        ("наивные полосы (B1)", naive.last_partition),
        ("Вороной+Ллойд без power", cvt.last_partition),
        ("Вороной+Ллойд+power (B1b)", power.last_partition),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12))

    draw_sectors(axes[0][0], naive.last_partition, f"Наивные полосы: N={N_AGENTS}")
    draw_tasks(axes[0][0], tasks, naive_assign)

    draw_sectors(
        axes[0][1], power.last_partition, f"Вороной + Ллойд + power: N={N_AGENTS}"
    )
    draw_tasks(axes[0][1], tasks, power_assign)

    draw_areas(axes[1][0], results)
    draw_shape(axes[1][1], results)

    fig.suptitle(
        f"Разбиение области на секторы — M1-1 (домен {AREA:.0f}×{AREA:.0f} м, "
        f"{len(tasks)} задач, {N_AGENTS} агентов)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "sectors_comparison.png"
    fig.savefig(path, dpi=110)
    print(f"Картинка сохранена: {path}")

    for (name, partition), assignment in zip(results, (naive_assign, None, power_assign)):
        stats = area_stats(partition)
        head = f"  {name:28s} площади: ratio={stats['ratio']:.3f}"
        if assignment is None:
            print(head)
            continue
        sizes = sorted(len(v) for v in assignment.by_agent.values())
        print(f"{head} | задач на агента: min={sizes[0]} max={sizes[-1]}")


def plot_plan() -> None:
    """Полный пайплайн: секторы → маршруты → проверка плана."""
    domain = box(0.0, 0.0, AREA, AREA)
    no_fly = (box(400, 400, 600, 600),)
    tasks = make_tasks()
    state = StubState(tuple(range(N_AGENTS)), domain)

    alloc = StaticVoronoiAllocator(lloyd_iters=50, power_iters=60)
    assignment = alloc.allocate(state, tasks)

    pads = {i: (50.0 + 90.0 * i, 20.0) for i in range(N_AGENTS)}
    routes = route_assignment(
        assignment,
        {t.id: t for t in tasks},
        positions=pads,
        homes=pads,
        budgets={i: 180_000.0 for i in range(N_AGENTS)},
    )

    validator = PlanValidator(
        no_fly=no_fly,
        domain=domain,
        min_sep=5.0,
        min_sep_vertical=3.0,
        z_range=(0.0, 120.0),
    )
    same = validator.validate(routes, altitudes={i: 50.0 for i in range(N_AGENTS)})
    layers = {i: 40.0 + 5.0 * i for i in range(N_AGENTS)}
    stacked = validator.validate(routes, altitudes=layers)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, report, title in (
        (axes[0], same, "все на 50 м"),
        (axes[1], stacked, "эшелонирование 40 + 5·i м"),
    ):
        draw_plan_panel(ax, alloc.last_partition, no_fly, routes, pads, report, title)

    fig.suptitle(
        f"План миссии: {N_AGENTS} агентов, {len(tasks)} задач, запретная зона в центре. "
        f"Нарушений: {len(same.violations)} → {len(stacked.violations)} "
        f"(высота снимает сепарацию, но не энергию и не обход зоны)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "plan_validation.png"
    fig.savefig(path, dpi=110)
    print(f"Картинка сохранена: {path}")
    print(f"  все на 50 м:     {same.by_rule()}")
    print(f"  эшелонирование:  {stacked.by_rule()}")


def draw_plan_panel(ax, partition, no_fly, routes, pads, report, title) -> None:
    cmap = plt.get_cmap("tab20")

    for sector in partition.sectors:
        if sector.is_empty:
            continue
        parts = sector.geoms if sector.geom_type == "MultiPolygon" else [sector]
        for g in parts:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="lightgray", linewidth=0.8, zorder=1)

    for zone in no_fly:
        xs, ys = zone.exterior.xy
        ax.add_patch(
            MplPolygon(
                list(zip(xs, ys)),
                closed=True,
                facecolor="#d62728",
                alpha=0.15,
                edgecolor="#d62728",
                linewidth=2,
                zorder=2,
            )
        )

    for agent, route in routes.items():
        ax.plot(
            [p[0] for p in route.waypoints],
            [p[1] for p in route.waypoints],
            color=cmap(agent % 20),
            linewidth=1.2,
            alpha=0.85,
            zorder=3,
        )

    ax.scatter(
        [pads[a][0] for a in pads],
        [pads[a][1] for a in pads],
        marker="^",
        s=45,
        color="black",
        zorder=5,
        label="площадки",
    )

    for v in report.violations:
        if v.location is not None:
            ax.scatter(*v.location, marker="x", s=70, color="#d62728", linewidths=2, zorder=6)
        if v.rule == "energy":
            for agent in v.agents:
                ax.scatter(
                    *pads[agent],
                    s=150,
                    facecolors="none",
                    edgecolors="#d62728",
                    linewidths=1.5,
                    zorder=6,
                )

    ax.set_title(f"{title}\nнарушений: {report.by_rule()}", fontsize=10)
    ax.set_xlim(0, AREA)
    ax.set_ylim(0, AREA)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2, linewidth=0.3)


if __name__ == "__main__":
    import sys

    if "--plan" in sys.argv[1:]:
        plot_plan()
    else:
        main()
