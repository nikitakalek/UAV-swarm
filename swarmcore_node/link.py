"""Канал «борт–борт» (PeerLink): UDP для стенда и борта, память — для тестов.

Узел отправляет каждое сообщение всем соседям и наземной станции (рассылка по списку
адресов: работает и в Wi-Fi, и в mesh-сети без широковещания). На MVP — Wi-Fi/UDP,
позже тот же интерфейс реализует промышленный mesh-радиомодуль.
"""

from __future__ import annotations

import random
import socket
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Optional

from swarmcore_node.messages import Msg

Addr = tuple[str, int]


class PeerLink(ABC):
    @abstractmethod
    def send(self, msg: Msg) -> None:
        """Отправить всем (соседям и наземной станции)."""

    @abstractmethod
    def recv(self) -> list[Msg]:
        """Все пришедшие сообщения (неблокирующе)."""

    def close(self) -> None:
        pass


class UdpLink(PeerLink):
    def __init__(self, bind: Addr, peers: Iterable[Addr]) -> None:
        self.peers = [p for p in peers if p != bind]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(bind)
        self.sock.setblocking(False)
        self.n_sent = self.n_recv = self.bytes_sent = 0

    def send(self, msg: Msg) -> None:
        data = msg.encode()
        for p in self.peers:
            try:
                self.sock.sendto(data, p)
                self.n_sent += 1
                self.bytes_sent += len(data)
            except OSError:
                pass  # сосед недоступен — это нормальная ситуация для роя

    def recv(self) -> list[Msg]:
        out = []
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            m = Msg.decode(data)
            if m is not None:
                out.append(m)
                self.n_recv += 1
        return out

    def close(self) -> None:
        self.sock.close()


class MemoryHub:
    """Общая «эфирная» среда для тестов: потери, разрывы связи, отключение участника."""

    def __init__(self, loss: float = 0.0, seed: int = 0) -> None:
        self.loss = loss
        self.rng = random.Random(seed)
        self.boxes: dict[int, deque[Msg]] = {}
        self.blocked: set[frozenset[int]] = set()  # пары, между которыми нет связи
        self.down: set[int] = set()  # участник не передаёт и не принимает

    def link(self, member: int) -> "MemoryLink":
        self.boxes.setdefault(member, deque())
        return MemoryLink(self, member)

    def cut(self, a: int, b: int) -> None:
        self.blocked.add(frozenset((a, b)))

    def heal(self) -> None:
        self.blocked.clear()

    def partition(self, *groups: Iterable[int]) -> None:
        gs = [set(g) for g in groups]
        for i, g in enumerate(gs):
            for h in gs[i + 1:]:
                for a in g:
                    for b in h:
                        self.cut(a, b)

    def _deliver(self, src: int, msg: Msg) -> None:
        if src in self.down:
            return
        for dst, box in self.boxes.items():
            if dst == src or dst in self.down or frozenset((src, dst)) in self.blocked:
                continue
            if self.loss and self.rng.random() < self.loss:
                continue
            box.append(Msg.decode(msg.encode()))  # копия через JSON — как по сети


class MemoryLink(PeerLink):
    def __init__(self, hub: MemoryHub, member: int) -> None:
        self.hub, self.member = hub, member

    def send(self, msg: Msg) -> None:
        self.hub._deliver(self.member, msg)

    def recv(self) -> list[Msg]:
        box = self.hub.boxes[self.member]
        out = list(box)
        box.clear()
        return out


def parse_addr(s: str, default_host: str = "127.0.0.1") -> Addr:
    host, _, port = s.rpartition(":")
    return (host or default_host, int(port))


def parse_addrs(s: Optional[str]) -> list[Addr]:
    return [parse_addr(x) for x in s.split(",") if x] if s else []
