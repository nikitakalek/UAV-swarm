"""Исполнение команд аппарату (arm, взлёт, посадка) без остановки цикла узла.

``MavsdkMotion.takeoff()`` ждёт окончания взлёта (до 90 с), ``land()`` — подтверждения
команды. Если вызывать их прямо из цикла узла, узел перестаёт слать heartbeat, и соседи
через ``peer_timeout`` считают его выбывшим. Поэтому для настоящего аппарата команды
идут в отдельном потоке (``ThreadCommands``), а узел проверяет результат на каждом шаге.

Для кинематической модели команды мгновенные — ``InlineCommands`` выполняет их сразу,
и тесты остаются детерминированными.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional


class InlineCommands:
    """Выполняет команду сразу в цикле узла."""

    def __init__(self) -> None:
        self._done: list[tuple[str, Optional[BaseException]]] = []

    @property
    def busy(self) -> bool:
        return False

    def submit(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            self._done.append((name, None))
        except Exception as e:  # noqa: BLE001 — ошибку аппарата узел обрабатывает сам
            self._done.append((name, e))

    def poll(self) -> list[tuple[str, Optional[BaseException]]]:
        out, self._done = self._done, []
        return out

    def run_now(self, fn: Callable[[], None]) -> None:
        fn()

    def close(self) -> None:
        pass


class ThreadCommands:
    """Команды по одной в фоновом потоке; результат забирается ``poll()`` из цикла узла."""

    def __init__(self, name: str = "vehicle") -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._pending: list[tuple[str, Future]] = []

    @property
    def busy(self) -> bool:
        return any(not f.done() for _, f in self._pending)

    def submit(self, name: str, fn: Callable[[], None]) -> None:
        self._pending.append((name, self._pool.submit(fn)))

    def poll(self) -> list[tuple[str, Optional[BaseException]]]:
        out = []
        while self._pending and self._pending[0][1].done():
            name, f = self._pending.pop(0)
            out.append((name, f.exception()))
        return out

    def run_now(self, fn: Callable[[], None]) -> None:
        """Срочная команда в обход очереди (имитация отказа), не дожидаясь результата."""
        threading.Thread(target=fn, daemon=True).start()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def make_commands(kind: str, name: str = "vehicle"):
    if kind == "thread":
        return ThreadCommands(name)
    if kind == "inline":
        return InlineCommands()
    raise ValueError(f"vehicle_commands: 'inline' или 'thread', а не {kind!r}")
