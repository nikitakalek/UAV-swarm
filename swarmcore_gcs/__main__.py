"""Наземная станция SwarmCore: ``python -m swarmcore_gcs [--port 8080]``."""

from __future__ import annotations

import argparse
import sys
import webbrowser

from swarmcore_gcs.server import make_server
from swarmcore_gcs.session import MissionSession


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="swarmcore_gcs", description="Пульт оператора и 3D-вид группы")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 — открыть пульт другим устройствам в сети")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--runs", default="runs/gcs", help="куда писать журналы миссий")
    p.add_argument("--node-port-base", type=int, default=47000)
    p.add_argument("--gcs-port", type=int, default=46999, help="UDP-порт станции для связи с узлами")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    session = MissionSession(args.runs, port_base=args.node_port_base, gcs_port=args.gcs_port)
    srv = make_server(args.host, args.port, session)
    base = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"Пульт оператора:  {base}/\n3D-вид:           {base}/view\nCtrl+C — выход", flush=True)
    if not args.no_browser:
        webbrowser.open(base + "/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
