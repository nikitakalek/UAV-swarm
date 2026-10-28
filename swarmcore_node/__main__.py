"""Запуск узлов и наземной станции.

    python -m swarmcore_node node --id 0 --n 3                       # один узел (по процессу на борт)
    python -m swarmcore_node gcs --mission scenarios/demo_field.yaml  # наземная станция: задача группе
    python -m swarmcore_node demo --fail 1@40 --time-scale 10         # всё сразу на одной машине + отчёт

Адреса по умолчанию — 127.0.0.1, порт узла ``--port-base + id``, станции ``--gcs-port``.
Для Raspberry Pi и SITL на другой машине: ``--peers host:port,...`` и ``--vehicle mavsdk``.

Часы: ``now = (time.time() − --epoch) · --time-scale``. Общая эпоха у всех процессов даёт
единую шкалу времени в журналах. Ускорение (``--time-scale`` > 1) — только для
кинематической модели; с PX4 SITL — 1.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from swarmcore_node.link import UdpLink, parse_addr, parse_addrs


def _clock(args):
    epoch = args.epoch if args.epoch is not None else time.time()
    return lambda: (time.time() - epoch) * args.time_scale


def _node_addrs(args) -> list[tuple[str, int]]:
    if args.peers:
        return parse_addrs(args.peers)
    return [(args.host, args.port_base + i) for i in range(args.n)]


def _make_motion(args):
    home = tuple(float(v) for v in args.home.split(",")) if args.home else (4.0 * args.id, 0.0)
    if args.vehicle == "mavsdk":
        from swarmcore_px4.mavsdk_motion import MavsdkMotion

        return MavsdkMotion(args.id, home, altitude=args.altitude, v_cruise=args.speed,
                            system_address=args.vehicle_address or None)
    from swarmsim.agents.kinematics import KinematicMotion

    return KinematicMotion(args.id, home, v_cruise=args.speed)


def cmd_node(args) -> int:
    from swarmcore_node.node import NodeConfig, SwarmNode

    addrs = _node_addrs(args)
    bind = ("0.0.0.0", addrs[args.id][1])
    link = UdpLink(bind, [a for i, a in enumerate(addrs) if i != args.id] + [parse_addr(args.gcs)])
    cfg = NodeConfig(battery_wh=args.battery_wh, peer_timeout=args.peer_timeout)
    node = SwarmNode(args.id, _make_motion(args), link, cfg)
    now = _clock(args)
    stop = {"flag": False}

    def _term(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    t_done = None
    try:
        while not stop["flag"]:
            node.step(now())
            if node.finished and t_done is None:
                t_done = now()
            if t_done is not None and now() - t_done > args.linger:
                break  # ещё немного шлём heartbeat «DONE», чтобы станция и соседи увидели
            time.sleep(args.loop_dt)
    finally:
        node.save(args.out)
        node.motion.close()
        link.close()
    return 0


def cmd_gcs(args) -> int:
    from swarmcore_node.gcs import GroundStation
    from swarmsim.scenario import load_scenario

    sc = load_scenario(args.mission)
    if args.n:
        sc.agents.count = args.n
    addrs = _node_addrs(argparse.Namespace(**{**vars(args), "n": sc.agents.count}))
    link = UdpLink(("0.0.0.0", parse_addr(args.gcs)[1]), addrs)
    gcs = GroundStation(link, sc)
    now = _clock(args)
    plan = sorted(_parse_at(args.fail, "fail") + _parse_at(args.rtl_at, "rtl") + _parse_at(args.silence, "silence"))
    last_print = -1e9
    t_end = args.timeout
    try:
        while now() < t_end:
            t = now()
            while plan and plan[0][0] <= t:
                _, what, arg = plan.pop(0)
                if what == "fail":
                    gcs.command(f"fail:{arg}")
                    if args.on_fail:
                        args.on_fail(int(arg))
                elif what == "rtl":
                    gcs.command("rtl")
                else:
                    gcs.silent = True
                    gcs.events.append({"t": round(t, 3), "type": "gcs_silent"})
            gcs.step(t)
            if not args.quiet and t - last_print >= args.print_every:
                last_print = t
                print(gcs.table(), flush=True)
            if gcs.mission_complete():
                print(gcs.table(), flush=True)
                print("миссия завершена", flush=True)
                break
            time.sleep(args.loop_dt)
    finally:
        gcs.save(args.out)
        link.close()
    return 0


def _parse_at(items, what):
    out = []
    for s in items or []:
        if "@" in s:
            arg, t = s.split("@")
        else:
            arg, t = "", s
        out.append((float(t), what, arg))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="swarmcore_node")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--n", type=int, default=3, help="число аппаратов в группе")
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--port-base", type=int, default=47000)
        sp.add_argument("--peers", help="адреса узлов host:port,... в порядке id (вместо host/port-base)")
        sp.add_argument("--gcs", default="127.0.0.1:46999", help="адрес наземной станции")
        sp.add_argument("--time-scale", type=float, default=1.0)
        sp.add_argument("--epoch", type=float, help="общая эпоха часов (unix time)")
        sp.add_argument("--loop-dt", type=float, default=0.05, help="период цикла, реальные секунды")
        sp.add_argument("--out", default="runs/node")

    sp = sub.add_parser("node", help="бортовой узел")
    common(sp)
    sp.add_argument("--id", type=int, required=True)
    sp.add_argument("--home", help="x,y точки старта, м (по умолчанию 4·id, 0)")
    sp.add_argument("--vehicle", choices=["kinematic", "mavsdk"], default="kinematic")
    sp.add_argument("--vehicle-address", help="адрес MAVSDK (по умолчанию udpin://0.0.0.0:14540+id)")
    sp.add_argument("--speed", type=float, default=5.0)
    sp.add_argument("--altitude", type=float, default=15.0, help="только для взлёта MAVSDK по умолчанию")
    sp.add_argument("--battery-wh", type=float, default=200.0)
    sp.add_argument("--peer-timeout", type=float, default=2.5)
    sp.add_argument("--linger", type=float, default=3.0)
    sp.set_defaults(func=cmd_node)

    sp = sub.add_parser("gcs", help="наземная станция")
    common(sp)
    sp.set_defaults(n=None)
    sp.add_argument("--mission", required=True)
    sp.add_argument("--fail", action="append", help="ID@T — имитировать отказ узла ID в момент T")
    sp.add_argument("--rtl-at", action="append", help="T — команда «все домой»")
    sp.add_argument("--silence", action="append", help="T — станция замолкает (потеря связи с оператором)")
    sp.add_argument("--timeout", type=float, default=3600.0)
    sp.add_argument("--print-every", type=float, default=10.0)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_gcs, on_fail=None)

    sp = sub.add_parser("demo", help="N узлов + станция на этой машине, отказ, отчёт и картинка")
    sp.add_argument("--mission", default=str(Path(__file__).resolve().parent.parent / "scenarios" / "demo_field.yaml"))
    sp.add_argument("--fail", default="1@40", help="ID@T; 'none' — без отказа")
    sp.add_argument("--fail-mode", choices=["kill", "command"], default="kill",
                    help="kill — убить процесс узла (как отказ борта); command — узел сам имитирует отказ")
    sp.add_argument("--time-scale", type=float, default=10.0)
    sp.add_argument("--port-base", type=int, default=47000)
    sp.add_argument("--gcs-port", type=int, default=46999)
    sp.add_argument("--out", default="runs/demo")
    sp.add_argument("--vehicle", choices=["kinematic", "mavsdk"], default="kinematic")
    sp.add_argument("--timeout", type=float, default=1800.0)
    sp.add_argument("--quiet", action="store_true")

    def _demo(a):
        from swarmcore_node.demo import run_demo

        return run_demo(a)

    sp.set_defaults(func=_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
