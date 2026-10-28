"""HTTP-сервер наземной станции: пульт (/), 3D-вид (/view), API и поток телеметрии.

    python -m swarmcore_gcs                 # http://127.0.0.1:8080 — пульт, /view — 3D

| метод | путь | что делает |
|---|---|---|
| GET | `/api/state` | состояние, геометрия текущей миссии |
| GET | `/api/stream` | SSE: снимок 4 раза в секунду + новые события ленты |
| POST | `/api/preview` | черновик → план: секторы, маршруты, время, замечания валидатора |
| POST | `/api/launch` | черновик → запуск узлов и миссии |
| POST | `/api/command` | `{"cmd": "rtl" | "rtl:1" | "pause" | "resume" | "fail:1" | "drain:1" | "gcs_link:off/on"}` |
| POST | `/api/mission/update` | в полёте: `{"nofly": [[lat, lon], ...]}` или `{"area": [...]}` |
| POST | `/api/stop` | остановить миссию |
| GET | `/tiles/<osm|sat>/z/x/y` | тайлы карты через кэш |
| GET | `/replay`, `/replay.json` | 3D-повтор последней завершённой миссии |
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from swarmcore_gcs.mission import DraftError, build, preview
from swarmcore_gcs.session import MissionSession
from swarmcore_gcs.tiles import get_tile

STATIC = Path(__file__).with_name("static")
PAGES = {"/": "gcs.html", "/gcs": "gcs.html", "/view": "view3d.html"}


class Handler(BaseHTTPRequestHandler):
    session: MissionSession  # задаётся в make_server
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # тише: только ошибки
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ ответы
    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html"))
                         else "max-age=86400")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _error(self, text: str, code: int = 400) -> None:
        self._json({"error": text}, code)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------------ маршруты
    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = url.path
        if path in PAGES:
            return self._send(200, (STATIC / PAGES[path]).read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            s = self.session
            return self._json({**s.snapshot(), "geometry": s.geometry()})
        if path == "/api/stream":
            return self._stream(int(parse_qs(url.query).get("feed", ["0"])[0]))
        if path.startswith("/tiles/"):
            parts = path.split("/")[2:]
            try:
                layer, z, x, y = parts[0], int(parts[1]), int(parts[2]), int(parts[3].split(".")[0])
            except (IndexError, ValueError):
                return self._error("неверный адрес тайла", 404)
            tile = get_tile(layer, z, x, y)
            if tile is None:
                return self._error("тайл недоступен", 404)
            return self._send(200, tile[0], tile[1])
        if path == "/replay":  # 3D-повтор последней миссии (страница из swarmcore_node, данные — /replay.json)
            from swarmcore_node.replay import TEMPLATE

            page = ('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" '
                    'content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
                    + TEMPLATE.read_text(encoding="utf-8") + "</body></html>")
            return self._send(200, page.encode(), "text/html; charset=utf-8")
        if path == "/replay.json":
            rd = self.session.run_dir
            if rd is None or not (rd / "replay.json").exists():
                return self._error("повтора ещё нет — миссия не завершена", 404)
            return self._send(200, (rd / "replay.json").read_bytes(), "application/json; charset=utf-8")
        self._error("не найдено", 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/preview":
                return self._json(preview(build(body)))
            if path == "/api/launch":
                return self._json(self.session.launch(build(body)))
            if path == "/api/command":
                self.session.command(str(body.get("cmd", "")))
                return self._json({"ok": True})
            if path == "/api/mission/update":
                return self._json({"rev": self.session.update_mission(body)})
            if path == "/api/stop":
                self.session.stop()
                return self._json({"ok": True})
        except (DraftError, ValueError, RuntimeError) as e:
            return self._error(str(e))
        self._error("не найдено", 404)

    def _stream(self, feed_from: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        n = feed_from
        try:
            while True:
                snap = self.session.snapshot(n)
                n = snap["feed_n"]
                self.wfile.write(b"data: " + json.dumps(snap, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError):
            return


def make_server(host: str = "127.0.0.1", port: int = 8080, session: MissionSession | None = None):
    handler = type("GcsHandler", (Handler,), {"session": session or MissionSession()})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv
