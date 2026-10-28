"""Отчёт по разбиению области на секторы (задача M1-1).

Печатает измеренные характеристики разбиения: разброс площадей, вытянутость
секторов, суммарный периметр. Нужен, чтобы «разброс площадей» был
воспроизводимой цифрой в отчёте, а не утверждением на словах.

Запуск:
    ./.venv/bin/python scripts/report.py
"""

from __future__ import annotations

from shapely.geometry import box

from swarmsim.sectors import (
    area_stats,
    build_partition,
    grid_seeds,
    lloyd_equalize,
    strip_partition,
)

AREA = 1000.0
N_VALUES = [5, 10, 20, 50]


def elongation(sector) -> float:
    """Изопериметрическое отношение: 1.0 у круга, ~1.13 у квадрата, много больше у полосы."""
    return sector.length / (4.0 * (sector.area ** 0.5))


def summarize(name: str, partition) -> None:
    st = area_stats(partition)
    q = [elongation(s) for s in partition.sectors if not s.is_empty]
    perimeter = float(sum(s.length for s in partition.sectors))
    print(
        f"  {name:26s} ratio={st['ratio']:5.3f}  cv={st['cv']:5.3f}  "
        f"вытянутость max={max(q):4.2f}  суммарный периметр={perimeter:9.0f}"
    )


def main() -> None:
    domain = box(0.0, 0.0, AREA, AREA)
    print(f"Домен: {AREA:.0f} x {AREA:.0f} м, площадь {domain.area:.0f} м²")
    print("ratio = max/min площадь сектора (1.0 = идеально равные); cv = коэффициент вариации\n")

    for n in N_VALUES:
        print(f"--- N = {n} агентов, идеальная площадь сектора {domain.area / n:.0f} м²")
        summarize("наивные полосы (B1)", strip_partition(domain, n))
        summarize(
            "Вороной + Ллойд (CVT)",
            lloyd_equalize(domain, grid_seeds(domain, n), iters=100),
        )
        summarize(
            "Вороной + Ллойд, hex",
            lloyd_equalize(domain, grid_seeds(domain, n, lattice="hex"), iters=100),
        )
        summarize(
            "Вороной + Ллойд + power (B1b)",
            build_partition(domain, n, lloyd_iters=100, power_iters=100),
        )
        print()


def report_allocation() -> None:
    """Замер T_compute и баланса загрузки для центрального аукциона (M1-5)."""
    from dataclasses import dataclass, field
    from time import perf_counter

    from swarmsim.alloc import CentralGreedyAllocator, StaticVoronoiAllocator

    @dataclass(frozen=True)
    class Task:
        id: int
        centroid: tuple[float, float]

    @dataclass(frozen=True)
    class State:
        agent_ids: tuple[int, ...]
        domain: object
        positions: dict = field(default_factory=dict)

        def domain_polygon(self):
            return self.domain

        def agent_position(self, agent_id: int):
            return self.positions[agent_id]

    dom = box(0.0, 0.0, AREA, AREA)
    tasks = [
        Task(i, (40 * ((i % 25) + 0.5), 40 * ((i // 25) + 0.5))) for i in range(625)
    ]

    print(f"Задач: {len(tasks)}. T_compute — время распределения, мс.\n")
    print(f"{'N':>3} | {'central, мс':>12} {'загрузка':>16} {'длина, км':>10} | {'voronoi, мс':>12}")
    for n in (5, 10, 20, 50):
        pads = {i: (50.0 + 900.0 * i / max(1, n - 1), 20.0) for i in range(n)}
        state = State(tuple(range(n)), dom, pads)

        start = perf_counter()
        central = CentralGreedyAllocator()
        assignment = central.allocate(state, tasks)
        t_central = (perf_counter() - start) * 1000

        start = perf_counter()
        StaticVoronoiAllocator(lloyd_iters=50, power_iters=60).allocate(state, tasks)
        t_voronoi = (perf_counter() - start) * 1000

        sizes = sorted(assignment.sizes().values())
        print(
            f"{n:>3} | {t_central:>12.1f} {min(sizes):>7}–{max(sizes):<8} "
            f"{central.last_cost / 1000:>10.1f} | {t_voronoi:>12.1f}"
        )


if __name__ == "__main__":
    import sys

    if "--alloc" in sys.argv[1:]:
        report_allocation()
    else:
        main()
