"""MavsdkMotion (M2-11): реализация ``MotionSource`` поверх MAVSDK (PX4 offboard).

Устройство:
* MAVSDK асинхронный, а ``MotionSource`` синхронный — поэтому у каждого аппарата
  свой цикл ``asyncio`` в фоновом потоке, публичные методы вызывают корутины через
  ``run_coroutine_threadsafe``;
* следование маршруту выполняется в фоне с частотой ``setpoint_hz`` (PX4 требует
  поток уставок offboard > 2 Гц), поэтому точность прохождения точек не зависит
  от шага симулятора ``dt``; ``step(dt)`` лишь возвращает свежее состояние;
* уставка — «морковка»: позиция на ``lookahead`` секунд впереди по направлению
  к точке + прямая подача скорости ``v_cruise`` (``set_position_velocity_ned``);
* система координат: x — восток, y — север (как в ``swarmsim``). Локальный NED
  каждого PX4 отсчитывается от его точки старта, поэтому мировая позиция =
  ``home + (east, north)``; точки старта в Gazebo задаются теми же ``home``.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Any, Optional, Sequence

from swarmsim.agents.base import MotionSource
from swarmsim.interfaces import State

BASE_UDP_PORT = 14540  # PX4 SITL: порт offboard API экземпляра i = 14540 + i
BASE_GRPC_PORT = 50051


def sitl_address(instance: int, host: str = "0.0.0.0") -> str:
    return f"udpin://{host}:{BASE_UDP_PORT + instance}"


class MavsdkMotion(MotionSource):
    def __init__(
        self,
        agent_id: int,
        home: tuple[float, float],
        altitude: float = 20.0,
        system_address: Optional[str] = None,
        grpc_port: Optional[int] = None,
        v_cruise: float = 5.0,
        accept_radius: float = 1.5,
        setpoint_hz: float = 20.0,
        lookahead_s: float = 1.5,
        failure_mode: str = "land",
        connect_timeout: float = 60.0,
        connect: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.home = (float(home[0]), float(home[1]))
        self.alt_target = altitude
        self.address = system_address or sitl_address(agent_id)
        self.grpc_port = grpc_port or (BASE_GRPC_PORT + agent_id)
        self.v_cruise = v_cruise
        self.accept_radius = accept_radius
        self.setpoint_period = 1.0 / setpoint_hz
        self.lookahead_s = lookahead_s
        self.failure_mode = failure_mode
        self.connect_timeout = connect_timeout

        self.t = 0.0
        self._ned = (0.0, 0.0, 0.0)  # north, east, down
        self._vel = (0.0, 0.0, 0.0)
        self._armed = False
        self._in_air = False
        self._flight_mode = "UNKNOWN"
        self._telemetry_t = 0.0
        self._telemetry_n = 0
        self._route: list[tuple[float, float]] = []
        self._idx = 0
        self._streaming = False
        self._killed = False
        self._lock = threading.Lock()
        self.events: list[tuple[float, str]] = []  # (monotonic, что произошло) — для отчёта о задержках

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=f"mavsdk-{agent_id}", daemon=True)
        self._thread.start()
        self.drone: Any = None
        if connect:
            self._call(self._connect(), timeout=connect_timeout + 5)

    # ------------------------------------------------------------------ инфраструктура
    def _call(self, coro, timeout: float = 30.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self) -> None:
        from mavsdk import System

        self.drone = System(port=self.grpc_port)
        await self.drone.connect(system_address=self.address)
        t0 = time.monotonic()
        async for st in self.drone.core.connection_state():
            if st.is_connected:
                break
            if time.monotonic() - t0 > self.connect_timeout:
                raise TimeoutError(f"агент {self.agent_id}: нет связи с {self.address}")
        await self.drone.telemetry.set_rate_position_velocity_ned(50.0)
        self._loop.create_task(self._watch_position())
        self._loop.create_task(self._watch_flags())
        async for h in self.drone.telemetry.health():
            if h.is_local_position_ok and h.is_global_position_ok and h.is_home_position_ok:
                break
            if time.monotonic() - t0 > self.connect_timeout:
                raise TimeoutError(f"агент {self.agent_id}: нет готовности навигации")
        self.events.append((time.monotonic(), "connected"))

    async def _watch_position(self) -> None:
        async for pv in self.drone.telemetry.position_velocity_ned():
            with self._lock:
                self._ned = (pv.position.north_m, pv.position.east_m, pv.position.down_m)
                self._vel = (pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s)
                self._telemetry_t = time.monotonic()
                self._telemetry_n += 1

    async def _watch_flags(self) -> None:
        async def armed():
            async for a in self.drone.telemetry.armed():
                self._armed = a

        async def in_air():
            async for a in self.drone.telemetry.in_air():
                self._in_air = a

        async def mode():
            async for m in self.drone.telemetry.flight_mode():
                self._flight_mode = str(m)

        await asyncio.gather(armed(), in_air(), mode())

    # ------------------------------------------------------------------ команды MotionSource
    def arm(self) -> None:
        async def go():
            for _ in range(20):
                try:
                    await self.drone.action.arm()
                    return
                except Exception:  # pre-arm ещё не прошёл
                    await asyncio.sleep(1.0)
            await self.drone.action.arm()

        self._call(go(), timeout=40)

    def takeoff(self, alt: float | None = None) -> None:
        alt = self.alt_target if alt is None else alt
        self.alt_target = alt

        async def go():
            await self.drone.action.set_takeoff_altitude(alt)
            await self.drone.action.takeoff()
            t0 = time.monotonic()
            while -self._ned[2] < 0.9 * alt:
                if time.monotonic() - t0 > 60:
                    raise TimeoutError(f"агент {self.agent_id}: взлёт не завершён")
                await asyncio.sleep(0.2)
            await self._start_offboard()

        self._call(go(), timeout=90)

    async def _start_offboard(self) -> None:
        from mavsdk.offboard import PositionNedYaw

        n, e, d = self._ned
        self._hold_point = (n, e, -self.alt_target)
        await self.drone.offboard.set_position_ned(PositionNedYaw(n, e, -self.alt_target, 0.0))
        await self.drone.offboard.start()
        self._streaming = True
        self._loop.create_task(self._stream())
        self.events.append((time.monotonic(), "offboard"))

    async def _stream(self) -> None:
        """Фоновое следование маршруту: продвижение по точкам и поток уставок."""
        from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

        while self._streaming and not self._killed:
            with self._lock:
                n, e, d = self._ned
                route, idx = self._route, self._idx
            while idx < len(route):
                tx, ty = route[idx]
                if math.hypot(tx - self.home[0] - e, ty - self.home[1] - n) <= self.accept_radius:
                    idx += 1
                else:
                    break
            with self._lock:
                if self._route is route:
                    self._idx = idx
            if idx < len(route):
                tx, ty = route[idx]
                tn, te = ty - self.home[1], tx - self.home[0]
                dn, de = tn - n, te - e
                dist = math.hypot(dn, de)
                # чистая уставка скорости: |v| ≤ v_cruise, торможение у точки, удержание высоты П-регулятором
                v = min(self.v_cruise, max(0.5, dist / self.lookahead_s))
                un, ue = dn / dist, de / dist
                vd = max(-1.5, min(1.5, 0.8 * (-self.alt_target - d)))
                yaw = math.degrees(math.atan2(ue, un))
                self._hold_point = (tn, te, -self.alt_target)
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(un * v, ue * v, vd, yaw))
            else:
                hn, he, hd = self._hold_point
                await self.drone.offboard.set_position_ned(PositionNedYaw(hn, he, hd, 0.0))
            await asyncio.sleep(self.setpoint_period)

    def set_route(self, points: Sequence[tuple[float, float]]) -> None:
        if self._killed:
            return
        with self._lock:
            self._route = [(float(x), float(y)) for x, y in points]
            self._idx = 0

    def hold(self) -> None:
        with self._lock:
            n, e, _ = self._ned
            self._route = []
            self._idx = 0
            self._hold_point = (n, e, -self.alt_target)

    def _stop_stream(self) -> None:
        self._streaming = False

        async def stop():
            try:
                await self.drone.offboard.stop()
            except Exception:
                pass

        self._call(stop(), timeout=10)

    def land(self) -> None:
        self._stop_stream()
        self._call(self.drone.action.land(), timeout=10)

    def rtl(self) -> None:
        self._stop_stream()
        self._call(self.drone.action.return_to_launch(), timeout=10)

    def kill(self) -> None:
        """Имитация отказа для испытаний: аппарат выходит из роя (посадка/зависание/kill)."""
        self._killed = True
        self._stop_stream()
        action = {"land": self.drone.action.land, "hold": self.drone.action.hold,
                  "kill": self.drone.action.kill}[self.failure_mode]
        self._call(action(), timeout=10)
        self.events.append((time.monotonic(), f"failure:{self.failure_mode}"))

    # ------------------------------------------------------------------ состояние
    def step(self, dt: float) -> State:
        self.t += dt
        return self.state()

    def state(self) -> State:
        with self._lock:
            n, e, d = self._ned
            vn, ve, _ = self._vel
            idx = self._idx
        mode = "failed" if self._killed else ("flying" if idx < len(self._route) else
                                             ("idle" if self._in_air else "landed"))
        return State(agent_id=self.agent_id, t=self.t, x=self.home[0] + e, y=self.home[1] + n, alt=-d,
                     vx=ve, vy=vn, armed=self._armed, mode=mode, route_index=idx)

    def telemetry_age(self) -> float:
        return time.monotonic() - self._telemetry_t

    # ------------------------------------------------------------------ задержка внешнего управления
    def measure_latency(self, trials: int = 10, speed: float = 2.0, threshold: float = 0.3) -> dict[str, Any]:
        """Задержка «уставка скорости → отклик в телеметрии» и RTT команд MAVLink.

        Аппарат должен висеть в offboard. На каждой попытке подаётся ступенька
        скорости ``speed`` на восток/запад; задержка — время до телеметрии, где
        скорость в этом направлении превысила ``threshold``.
        """
        from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

        async def go():
            self._streaming = False
            await asyncio.sleep(0.2)
            step_lat, rtt = [], []
            for k in range(trials):
                sign = 1.0 if k % 2 == 0 else -1.0
                n, e, d = self._ned
                t0 = time.monotonic()
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, sign * speed, 0.0, 0.0))
                rtt.append(time.monotonic() - t0)
                while True:
                    await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, sign * speed, 0.0, 0.0))
                    if sign * self._vel[1] > threshold:
                        step_lat.append(time.monotonic() - t0)
                        break
                    if time.monotonic() - t0 > 5:
                        break
                    await asyncio.sleep(0.005)
                for _ in range(40):  # стабилизация на месте
                    await self.drone.offboard.set_position_ned(PositionNedYaw(n, e, -self.alt_target, 0.0))
                    await asyncio.sleep(0.05)
            c0, n0 = time.monotonic(), self._telemetry_n
            await asyncio.sleep(2.0)
            rate = (self._telemetry_n - n0) / (time.monotonic() - c0)
            p0 = time.monotonic()
            for _ in range(5):
                await self.drone.action.get_takeoff_altitude()
            param_rtt = (time.monotonic() - p0) / 5
            self._streaming = True
            self._loop.create_task(self._stream())
            return step_lat, rtt, rate, param_rtt

        step_lat, rtt, rate, param_rtt = self._call(go(), timeout=trials * 10 + 30)
        s = sorted(step_lat)
        return {
            "agent": self.agent_id,
            "trials": trials,
            "step_response_ms_median": 1000 * s[len(s) // 2] if s else None,
            "step_response_ms_p90": 1000 * s[int(0.9 * (len(s) - 1))] if s else None,
            "setpoint_send_ms_median": 1000 * sorted(rtt)[len(rtt) // 2],
            "command_rtt_ms": 1000 * param_rtt,
            "telemetry_hz": rate,
        }

    def close(self) -> None:
        try:
            if self._streaming and not self._killed:
                self.land()
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
