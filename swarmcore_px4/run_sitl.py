"""Прогон сценария SwarmCore на PX4 SITL (M2-11 / M2-12) и служебные проверки.

    # 1) поднять PX4 SITL (в WSL/Linux):
    HOMES="0,0 4,0 8,0" swarmcore_px4/sitl/multi_sitl.sh 3
    # 2) проверить уникальность sysid/портов и одновременный полёт:
    python -m swarmcore_px4.run_sitl check --n 3
    # 3) измерить задержку внешнего управления:
    python -m swarmcore_px4.run_sitl latency --n 1
    # 4) прогнать сценарий тем же SimLoop, но с MavsdkMotion вместо KinematicMotion:
    python -m swarmcore_px4.run_sitl run --scenario scenarios/sitl_small.yaml --out runs/sitl

Логика роя и цикл симуляции не меняются: подменяется только ``motion_factory``.
Цикл идёт в реальном времени (PX4 SITL — real-time), поэтому настенное время
здесь влияет на результат — это ожидаемо и отражено в manifest (``mode: sitl``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _motion_factory(args, speed_override=None):
    from swarmcore_px4.mavsdk_motion import MavsdkMotion, sitl_address

    def make(i, home, alt, cfg):
        return MavsdkMotion(i, home, altitude=alt, system_address=sitl_address(i + args.first_instance),
                            grpc_port=50051 + i + args.first_instance,
                            v_cruise=speed_override or cfg.speed, accept_radius=args.accept_radius,
                            failure_mode=args.failure_mode, connect=True)

    return make


def cmd_run(args) -> int:
    from swarmsim.scenario import Scenario
    from swarmsim.sim import Simulation

    sc = Scenario.from_yaml(args.scenario)
    print(f"сценарий {sc.name}: N={sc.agents.count}, задач по сетке {sc.mission.n_cells}; подключение к PX4…")
    factory = _motion_factory(args)
    t_connect = time.monotonic()

    # подключение и взлёт параллельно, затем — обычный Simulation с готовыми источниками движения
    def prepare(i):
        home = sc.agents.home_of(i)
        alt = sc.agents.altitude + i * sc.agents.altitude_step
        m = factory(i, home, alt, sc.agents)
        m.arm()
        m.takeoff(alt)
        return m

    with ThreadPoolExecutor(max_workers=sc.agents.count) as ex:
        ready = list(ex.map(prepare, range(sc.agents.count)))
    print(f"все {len(ready)} аппарата в воздухе за {time.monotonic() - t_connect:.1f} с; старт миссии")

    class _Prepared:
        """Отдаёт уже взлетевшие источники; arm/takeoff в Simulation — без повторного взлёта."""

        def __init__(self, motions):
            self.motions = motions

        def __call__(self, i, home, alt, cfg):
            m = self.motions[i]
            m.arm = lambda: None
            m.takeoff = lambda alt=None: None
            return m

    sim = Simulation(sc, args.seed, motion_factory=_Prepared(ready))
    wall0 = time.monotonic()
    last_print = [0.0]

    def pace(s):
        target = wall0 + s.t / args.speed
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if s.t - last_print[0] >= 10:
            last_print[0] = s.t
            done = sum(t.status.value == "done" for t in s.tasks)
            pos = " ".join(f"A{i}:({a.state.x:.0f},{a.state.y:.0f},{a.state.alt:.0f}){'✗' if not a.alive else ''}"
                           for i, a in s.agents.items())
            print(f"t={s.t:6.1f} задач {done}/{len(s.tasks)} lag={-delay:+.2f}s  {pos}", flush=True)

    res = sim.run(pace=pace)
    res.manifest.update({"mode": "sitl", "realtime_speed": args.speed,
                         "wall_s": time.monotonic() - wall0, "px4_instances": [args.first_instance + i
                                                                                for i in range(sc.agents.count)]})
    out = res.save(Path(args.out) / f"{sc.name}__sitl" / f"seed_{args.seed:03d}")
    m = res.metrics
    print(f"\nготово → {out}\nCR={m['CR']:.2f} makespan={m['makespan']:.1f} с T_recovery={m['T_recovery_mean']} "
          f"N_dup={m['N_dup']} N_lost={m['N_lost']} valid={m['valid']}")
    if args.compare_kinematic:
        from swarmsim.sim import run as run_kin

        k = run_kin(sc, args.seed)
        k.save(Path(args.out) / f"{sc.name}__kinematic" / f"seed_{args.seed:03d}")
        print(f"кинематика: CR={k.metrics['CR']:.2f} makespan={k.metrics['makespan']:.1f} с → "
              f"SITL/кинематика = {m['makespan'] / k.metrics['makespan']:.2f}")
    return 0 if m["valid"] else 1


def cmd_check(args) -> int:
    """M2-12: уникальные MAV_SYS_ID и порты, одновременный полёт без конфликтов."""
    from swarmcore_px4.mavsdk_motion import MavsdkMotion, sitl_address

    motions = []
    for i in range(args.n):
        inst = args.first_instance + i
        m = MavsdkMotion(i, (args.spacing * i, 0.0), altitude=10.0 + 3 * i, system_address=sitl_address(inst),
                         grpc_port=50051 + inst, v_cruise=4.0)
        sysid = m._call(m.drone.param.get_param_int("MAV_SYS_ID"))
        print(f"экземпляр {inst}: порт {14540 + inst}, MAV_SYS_ID={sysid}")
        motions.append((m, sysid))
    ids = [s for _, s in motions]
    ok = len(set(ids)) == len(ids)
    print("sysid уникальны" if ok else f"КОНФЛИКТ sysid: {ids}")
    with ThreadPoolExecutor(max_workers=args.n) as ex:
        list(ex.map(lambda mm: (mm[0].arm(), mm[0].takeoff()), motions))
    # одновременный полёт: квадрат 15×15 м у каждого
    for m, _ in motions:
        hx, hy = m.home
        m.set_route([(hx, hy + 15), (hx + 15, hy + 15), (hx + 15, hy), (hx, hy)])
    t0 = time.monotonic()
    min_sep = math.inf
    v_peak = 0.0
    while time.monotonic() - t0 < 120:
        sts = [m.state() for m, _ in motions]
        v_peak = max([v_peak] + [s.speed for s in sts])
        for a in range(len(sts)):
            for b in range(a + 1, len(sts)):
                d = math.dist((sts[a].x, sts[a].y, sts[a].alt), (sts[b].x, sts[b].y, sts[b].alt))
                min_sep = min(min_sep, d)
        if all(s.route_index >= 4 for s in sts):
            break
        time.sleep(0.2)
    done = [m.state().route_index >= 4 for m, _ in motions]
    print(f"маршруты пройдены: {done} за {time.monotonic() - t0:.1f} с; мин. дистанция между бортами {min_sep:.1f} м; "
          f"пиковая горизонтальная скорость {v_peak:.2f} м/с (v_cruise 4.0)")
    for m, _ in motions:
        m.close()
    ok = ok and all(done)
    print("ПРОВЕРКА ПРОЙДЕНА" if ok else "ПРОВЕРКА НЕ ПРОЙДЕНА")
    return 0 if ok else 1


def cmd_latency(args) -> int:
    from swarmcore_px4.mavsdk_motion import MavsdkMotion, sitl_address

    rows = []
    motions = [MavsdkMotion(i, (0.0, 0.0), altitude=10.0, system_address=sitl_address(args.first_instance + i),
                            grpc_port=50051 + args.first_instance + i) for i in range(args.n)]
    for m in motions:
        m.arm()
        m.takeoff()
    for m in motions:
        r = m.measure_latency(trials=args.trials)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    for m in motions:
        m.close()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="swarmcore_px4.run_sitl")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--first-instance", type=int, default=0)
    common.add_argument("--accept-radius", type=float, default=1.5)
    common.add_argument("--failure-mode", choices=["land", "hold", "kill"], default="land")

    r = sub.add_parser("run", parents=[common])
    r.add_argument("--scenario", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", default="runs/sitl")
    r.add_argument("--speed", type=float, default=1.0, help="коэффициент скорости SITL (PX4_SIM_SPEED_FACTOR)")
    r.add_argument("--compare-kinematic", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("check", parents=[common])
    c.add_argument("--n", type=int, default=2)
    c.add_argument("--spacing", type=float, default=4.0)
    c.set_defaults(func=cmd_check)

    lt = sub.add_parser("latency", parents=[common])
    lt.add_argument("--n", type=int, default=1)
    lt.add_argument("--trials", type=int, default=10)
    lt.add_argument("--out")
    lt.set_defaults(func=cmd_latency)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
