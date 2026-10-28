"""Тайлы карты через сервер пульта: кэш на диске и один источник для карты и 3D-сцены.

Через свой сервер тайлы не упираются в CORS, когда 3D-страница рисует их на текстуре,
а уже просмотренные места открываются без интернета.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

SOURCES = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "sat": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}
USER_AGENT = "SwarmCore-GCS/0.1 (student project, MAI)"
CACHE = Path.home() / ".cache" / "swarmcore" / "tiles"


def get_tile(layer: str, z: int, x: int, y: int) -> Optional[tuple[bytes, str]]:
    if layer not in SOURCES or not (0 <= z <= 19):
        return None
    path = CACHE / layer / str(z) / str(x) / f"{y}.img"
    if path.exists():
        data = path.read_bytes()
    else:
        req = urllib.request.Request(SOURCES[layer].format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
        except OSError:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    ctype = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return data, ctype
